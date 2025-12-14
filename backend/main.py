"""FastAPI backend for passport photo tool."""

import base64
import io
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from .config import PHOTO_SPECS, OutputFormat
from .image_processor import ImageProcessor

app = FastAPI(
    title="Passport Photo Tool",
    description="Tool for resizing and processing photos for Indian passport applications",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize image processor
processor = ImageProcessor()


class PhotoSpecResponse(BaseModel):
    """Response model for photo specifications."""
    name: str
    width_mm: float
    height_mm: float
    width_px: int
    height_px: int
    min_file_size_kb: int
    max_file_size_kb: int
    dpi: int
    aspect_ratio: float


class FaceDetectionResponse(BaseModel):
    """Response model for face detection."""
    detected: bool
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


class ValidationResponse(BaseModel):
    """Response model for validation results."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


class ProcessResponse(BaseModel):
    """Response model for processed photo."""
    success: bool
    message: str
    image_base64: str | None = None
    face_detection: FaceDetectionResponse | None = None
    validation: ValidationResponse | None = None


class AnalyzeResponse(BaseModel):
    """Response model for image analysis."""
    success: bool
    message: str
    width: int
    height: int
    face_detection: FaceDetectionResponse | None = None
    quality_score: float
    quality_issues: list[str]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Passport Photo Tool API"}


@app.get("/specs", response_model=dict[str, PhotoSpecResponse])
async def get_specs():
    """Get photo specifications for all output formats."""
    return {
        fmt.value: PhotoSpecResponse(
            name=spec.name,
            width_mm=spec.width_mm,
            height_mm=spec.height_mm,
            width_px=spec.width_px,
            height_px=spec.height_px,
            min_file_size_kb=spec.min_file_size_kb,
            max_file_size_kb=spec.max_file_size_kb,
            dpi=spec.dpi,
            aspect_ratio=spec.aspect_ratio
        )
        for fmt, spec in PHOTO_SPECS.items()
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded image without processing it."""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")

        # Detect face
        face = processor.detect_face(image)

        # Check quality
        quality_score, quality_issues = processor.check_image_quality(image)

        face_response = None
        if face.detected:
            face_response = FaceDetectionResponse(
                detected=True,
                x=face.x,
                y=face.y,
                width=face.width,
                height=face.height
            )

        return AnalyzeResponse(
            success=True,
            message="Image analyzed successfully",
            width=image.width,
            height=image.height,
            face_detection=face_response,
            quality_score=quality_score,
            quality_issues=quality_issues
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/process", response_model=ProcessResponse)
async def process_image(
    file: UploadFile = File(...),
    output_format: Annotated[str, Form()] = "print",
    remove_background: Annotated[bool, Form()] = True,
    auto_crop: Annotated[bool, Form()] = True,
    crop_x: Annotated[int | None, Form()] = None,
    crop_y: Annotated[int | None, Form()] = None,
    crop_width: Annotated[int | None, Form()] = None,
    crop_height: Annotated[int | None, Form()] = None,
    brightness: Annotated[int, Form()] = 100,
    contrast: Annotated[int, Form()] = 100,
):
    """Process an uploaded image for passport photo requirements."""
    try:
        # Parse output format
        try:
            fmt = OutputFormat(output_format)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid output format. Must be one of: {[f.value for f in OutputFormat]}"
            )

        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")

        # Apply brightness/contrast adjustments
        if brightness != 100 or contrast != 100:
            from PIL import ImageEnhance
            if brightness != 100:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness / 100)
            if contrast != 100:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast / 100)

        # Build manual crop box if provided
        manual_crop_box = None
        if all(v is not None for v in [crop_x, crop_y, crop_width, crop_height]):
            manual_crop_box = (
                crop_x,
                crop_y,
                crop_x + crop_width,
                crop_y + crop_height
            )

        # Process the image
        result = processor.process_photo(
            image=image,
            output_format=fmt,
            remove_bg=remove_background,
            auto_crop=auto_crop,
            manual_crop_box=manual_crop_box
        )

        if not result.success:
            return ProcessResponse(
                success=False,
                message=result.message
            )

        # Convert processed image to base64
        spec = PHOTO_SPECS[fmt]
        image_bytes = processor.optimize_for_file_size(result.image, spec)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Build response
        face_response = None
        if result.face_detection and result.face_detection.detected:
            face_response = FaceDetectionResponse(
                detected=True,
                x=result.face_detection.x,
                y=result.face_detection.y,
                width=result.face_detection.width,
                height=result.face_detection.height
            )

        validation_response = None
        if result.validation:
            validation_response = ValidationResponse(
                is_valid=result.validation.is_valid,
                errors=result.validation.errors,
                warnings=result.validation.warnings
            )

        return ProcessResponse(
            success=True,
            message=result.message,
            image_base64=image_base64,
            face_detection=face_response,
            validation=validation_response
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download")
async def download_image(
    file: UploadFile = File(...),
    output_format: Annotated[str, Form()] = "print",
    remove_background: Annotated[bool, Form()] = True,
    auto_crop: Annotated[bool, Form()] = True,
    image_format: Annotated[str, Form()] = "jpeg",
):
    """Process and download the image directly."""
    try:
        # Parse output format
        try:
            fmt = OutputFormat(output_format)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid output format. Must be one of: {[f.value for f in OutputFormat]}"
            )

        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")

        # Process the image
        result = processor.process_photo(
            image=image,
            output_format=fmt,
            remove_bg=remove_background,
            auto_crop=auto_crop
        )

        if not result.success or result.image is None:
            raise HTTPException(status_code=400, detail=result.message)

        # Optimize for file size
        spec = PHOTO_SPECS[fmt]
        image_bytes = processor.optimize_for_file_size(
            result.image,
            spec,
            target_format=image_format.upper()
        )

        # Determine content type
        content_type = "image/jpeg" if image_format.lower() == "jpeg" else f"image/{image_format.lower()}"

        return Response(
            content=image_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=passport_photo.{image_format.lower()}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
