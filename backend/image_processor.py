"""Image processing logic for passport photos."""

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from rembg import remove

from .config import PHOTO_SPECS, OutputFormat, PhotoSpec


@dataclass
class FaceDetectionResult:
    """Result of face detection."""
    detected: bool
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 0.0
    eye_left: tuple[int, int] | None = None
    eye_right: tuple[int, int] | None = None


@dataclass
class ValidationResult:
    """Result of photo validation."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class ProcessingResult:
    """Result of image processing."""
    success: bool
    image: Image.Image | None
    face_detection: FaceDetectionResult | None
    validation: ValidationResult | None
    message: str


class ImageProcessor:
    """Handles all image processing operations for passport photos."""

    def __init__(self):
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def detect_face(self, image: Image.Image) -> FaceDetectionResult:
        """Detect face in the image using OpenCV."""
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        if len(faces) == 0:
            return FaceDetectionResult(detected=False)

        # Take the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # Detect eyes within the face region
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray)

        eye_left = None
        eye_right = None
        if len(eyes) >= 2:
            # Sort eyes by x position
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            eye_left = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
            eye_right = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)

        return FaceDetectionResult(
            detected=True,
            x=int(x),
            y=int(y),
            width=int(w),
            height=int(h),
            confidence=0.9,  # Haar cascades don't provide confidence
            eye_left=eye_left,
            eye_right=eye_right
        )

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background and replace with white."""
        # Use rembg to remove background
        output = remove(image)

        # Create white background
        white_bg = Image.new('RGBA', output.size, (255, 255, 255, 255))

        # Composite the image onto white background
        white_bg.paste(output, mask=output.split()[3] if output.mode == 'RGBA' else None)

        # Convert back to RGB
        return white_bg.convert('RGB')

    def auto_crop_to_face(
        self,
        image: Image.Image,
        face: FaceDetectionResult,
        spec: PhotoSpec
    ) -> Image.Image:
        """Automatically crop image to center on face with proper proportions."""
        img_width, img_height = image.size

        # Calculate the ideal crop based on face position
        # Face should occupy 70-80% of the frame height
        target_face_ratio = 0.70

        # Calculate crop dimensions based on aspect ratio
        ideal_crop_height = face.height / target_face_ratio
        ideal_crop_width = ideal_crop_height * spec.aspect_ratio

        # Center the crop on the face
        face_center_x = face.x + face.width // 2
        face_center_y = face.y + face.height // 2

        # For passport photos, we want more space above the head
        # Adjust vertical center to be slightly below face center
        adjusted_center_y = face_center_y + face.height * 0.1

        # Calculate crop boundaries
        left = face_center_x - ideal_crop_width // 2
        top = adjusted_center_y - ideal_crop_height // 2
        right = left + ideal_crop_width
        bottom = top + ideal_crop_height

        # Ensure we don't go outside image bounds
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > img_width:
            left -= (right - img_width)
            right = img_width
        if bottom > img_height:
            top -= (bottom - img_height)
            bottom = img_height

        # Clamp to image bounds
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        # Crop the image
        cropped = image.crop((int(left), int(top), int(right), int(bottom)))

        # Resize to target dimensions
        return cropped.resize((spec.width_px, spec.height_px), Image.Resampling.LANCZOS)

    def manual_crop(
        self,
        image: Image.Image,
        crop_box: tuple[int, int, int, int],
        spec: PhotoSpec
    ) -> Image.Image:
        """Manually crop image with given coordinates and resize to spec."""
        cropped = image.crop(crop_box)
        return cropped.resize((spec.width_px, spec.height_px), Image.Resampling.LANCZOS)

    def validate_photo(
        self,
        image: Image.Image,
        face: FaceDetectionResult,
        spec: PhotoSpec
    ) -> ValidationResult:
        """Validate photo against passport requirements."""
        errors = []
        warnings = []

        img_width, img_height = image.size

        # Check if face was detected
        if not face.detected:
            errors.append("No face detected in the image")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check face size relative to image
        face_height_percent = (face.height / img_height) * 100
        if face_height_percent < spec.min_face_height_percent:
            errors.append(
                f"Face is too small ({face_height_percent:.1f}%). "
                f"Should be at least {spec.min_face_height_percent}% of image height."
            )
        elif face_height_percent > spec.max_face_height_percent:
            errors.append(
                f"Face is too large ({face_height_percent:.1f}%). "
                f"Should be at most {spec.max_face_height_percent}% of image height."
            )

        # Check eye position
        if face.eye_left and face.eye_right:
            avg_eye_y = (face.eye_left[1] + face.eye_right[1]) / 2
            eye_position_from_bottom = img_height - avg_eye_y
            eye_position_percent = (eye_position_from_bottom / img_height) * 100

            if eye_position_percent < spec.min_eye_position_percent:
                warnings.append(
                    f"Eyes may be positioned too low ({eye_position_percent:.1f}% from bottom)"
                )
            elif eye_position_percent > spec.max_eye_position_percent:
                warnings.append(
                    f"Eyes may be positioned too high ({eye_position_percent:.1f}% from bottom)"
                )
        else:
            warnings.append("Could not detect eyes for precise positioning validation")

        # Check image dimensions match expected aspect ratio
        actual_ratio = img_width / img_height
        expected_ratio = spec.aspect_ratio
        if abs(actual_ratio - expected_ratio) > 0.05:
            warnings.append(
                f"Image aspect ratio ({actual_ratio:.2f}) differs from expected ({expected_ratio:.2f})."
            )

        # Check for sufficient resolution
        if img_width < 350 or img_height < 350:
            errors.append(
                f"Image resolution too low ({img_width}x{img_height}). "
                "Minimum 350x350 pixels required."
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def check_image_quality(self, image: Image.Image) -> tuple[float, list[str]]:
        """Check image quality (sharpness, brightness, etc.)."""
        issues = []

        # Convert to grayscale for analysis
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Check sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            issues.append("Image appears blurry. Please use a sharper photo.")

        # Check brightness
        mean_brightness = np.mean(gray)
        if mean_brightness < 80:
            issues.append("Image appears too dark. Please use better lighting.")
        elif mean_brightness > 220:
            issues.append("Image appears overexposed. Please reduce lighting.")

        # Calculate quality score (0-1)
        sharpness_score = min(1.0, laplacian_var / 500)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        quality_score = (sharpness_score + brightness_score) / 2

        return quality_score, issues

    def optimize_for_file_size(
        self,
        image: Image.Image,
        spec: PhotoSpec,
        target_format: str = "JPEG"
    ) -> bytes:
        """Optimize image to meet file size requirements."""
        buffer = io.BytesIO()

        # Start with high quality and reduce if needed
        quality = 95

        while quality > 20:
            buffer.seek(0)
            buffer.truncate()

            if target_format.upper() == "JPEG":
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
            else:
                image.save(buffer, format=target_format, optimize=True)

            size_kb = buffer.tell() / 1024

            if size_kb <= spec.max_file_size_kb:
                break

            quality -= 5

        return buffer.getvalue()

    def process_photo(
        self,
        image: Image.Image,
        output_format: OutputFormat,
        remove_bg: bool = True,
        auto_crop: bool = True,
        manual_crop_box: tuple[int, int, int, int] | None = None
    ) -> ProcessingResult:
        """Main processing pipeline for passport photos."""
        spec = PHOTO_SPECS[output_format]

        try:
            # Step 1: Detect face in original image
            face = self.detect_face(image)

            if not face.detected:
                return ProcessingResult(
                    success=False,
                    image=None,
                    face_detection=face,
                    validation=None,
                    message="No face detected in the image. Please upload a clear frontal photo."
                )

            # Step 2: Remove background if requested
            if remove_bg:
                image = self.remove_background(image)

            # Step 3: Crop image
            if manual_crop_box:
                image = self.manual_crop(image, manual_crop_box, spec)
            elif auto_crop:
                image = self.auto_crop_to_face(image, face, spec)
            else:
                # Just resize to target dimensions
                image = image.resize(
                    (spec.width_px, spec.height_px),
                    Image.Resampling.LANCZOS
                )

            # Step 4: Re-detect face in processed image
            processed_face = self.detect_face(image)

            # Step 5: Validate the processed photo
            validation = self.validate_photo(image, processed_face, spec)

            # Step 6: Check image quality
            quality_score, quality_issues = self.check_image_quality(image)
            validation.warnings.extend(quality_issues)

            return ProcessingResult(
                success=True,
                image=image,
                face_detection=processed_face,
                validation=validation,
                message="Photo processed successfully" if validation.is_valid
                        else "Photo processed with issues"
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                image=None,
                face_detection=None,
                validation=None,
                message=f"Error processing image: {str(e)}"
            )
