"""Configuration for passport photo specifications."""

from dataclasses import dataclass
from enum import Enum


class OutputFormat(str, Enum):
    PRINT_SQUARE = "print_square"  # 51x51mm (2x2 inch) for US-style printing
    PRINT_RECT = "print_rect"  # 35x45mm (3.5x4.5cm) standard passport size
    DIGITAL = "digital"  # 630x810px for online submission


@dataclass
class PhotoSpec:
    """Photo specification for a given format."""
    name: str
    width_mm: float
    height_mm: float
    width_px: int
    height_px: int
    min_file_size_kb: int
    max_file_size_kb: int
    dpi: int
    background_color: tuple[int, int, int]  # RGB
    aspect_ratio: float  # width/height

    # Face positioning requirements (as percentage of image height)
    min_face_height_percent: float
    max_face_height_percent: float
    min_eye_position_percent: float  # from bottom
    max_eye_position_percent: float  # from bottom


# Indian Passport Photo Specifications
# Sources:
# - https://visafoto.com/in-passport-35x45mm-photo
# - https://www.blsinternational.com/india/uae/passport/passport-photo-specification.php
# - https://passport-photo.online/indian-passport-photo

PHOTO_SPECS = {
    OutputFormat.PRINT_SQUARE: PhotoSpec(
        name="Print Square (51×51mm / 2×2 inch)",
        width_mm=51.0,
        height_mm=51.0,
        width_px=600,  # At 300 DPI
        height_px=600,
        min_file_size_kb=10,
        max_file_size_kb=1000,
        dpi=300,
        background_color=(255, 255, 255),  # White
        aspect_ratio=1.0,
        min_face_height_percent=50,
        max_face_height_percent=70,
        min_eye_position_percent=56,  # 28.5mm from bottom / 51mm
        max_eye_position_percent=69,  # 35mm from bottom / 51mm
    ),
    OutputFormat.PRINT_RECT: PhotoSpec(
        name="Print Standard (35×45mm / 3.5×4.5cm)",
        width_mm=35.0,
        height_mm=45.0,
        width_px=413,  # At 300 DPI (35mm = 1.378 inch * 300)
        height_px=531,  # At 300 DPI (45mm = 1.772 inch * 300)
        min_file_size_kb=10,
        max_file_size_kb=1000,
        dpi=300,
        background_color=(255, 255, 255),  # White
        aspect_ratio=35.0 / 45.0,  # ~0.778
        min_face_height_percent=60,
        max_face_height_percent=80,
        min_eye_position_percent=50,
        max_eye_position_percent=70,
    ),
    OutputFormat.DIGITAL: PhotoSpec(
        name="Digital Upload (630×810px)",
        width_mm=35.0,  # Corresponds to 35x45mm at screen resolution
        height_mm=45.0,
        width_px=630,  # Official requirement for online submission
        height_px=810,
        min_file_size_kb=20,
        max_file_size_kb=100,
        dpi=200,  # Lower DPI for digital
        background_color=(255, 255, 255),  # White
        aspect_ratio=630.0 / 810.0,  # ~0.778
        min_face_height_percent=60,
        max_face_height_percent=80,
        min_eye_position_percent=50,
        max_eye_position_percent=70,
    ),
}

# Validation thresholds
FACE_DETECTION_CONFIDENCE = 0.5
MIN_IMAGE_QUALITY = 0.7  # Minimum sharpness score
