"""Configuration for Indian passport photo specifications.

Official sources:
- MEA Passport Seva: https://www.passportindia.gov.in/
- VFS Global USA: https://visa.vfsglobal.com/usa/en/ind/
- ICAO Document 9303 (Machine Readable Travel Documents)

Last updated: 2024 (includes digital submission requirements)
"""

from dataclasses import dataclass, field
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

    # Additional validation parameters
    min_head_top_margin_percent: float = 5.0  # Space above head
    max_head_top_margin_percent: float = 15.0

    # Background tolerance (how close to white is acceptable)
    background_tolerance: int = 30  # RGB deviation from white allowed

    # Minimum resolution for source image
    min_source_resolution: int = 350


# Indian Passport Photo Specifications
# Based on MEA guidelines and ICAO standards (2024)
#
# Key requirements:
# - White or off-white background (no shadows)
# - Neutral expression, mouth closed
# - Face clearly visible from forehead to chin
# - Both edges of face visible
# - No dark/tinted glasses (regular glasses OK without glare)
# - Head straight, not tilted
# - Even lighting, no shadows on face

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
        # Head should be 25-35mm (50-69% of 51mm)
        # Note: OpenCV detects "face" (forehead to chin) not full head
        # Actual face detection is typically smaller than full head
        min_face_height_percent=40,
        max_face_height_percent=75,
        # Eyes should be 28-35mm from bottom (55-69% of 51mm)
        min_eye_position_percent=45,
        max_eye_position_percent=75,
        min_head_top_margin_percent=3.0,
        max_head_top_margin_percent=25.0,
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
        # Relaxed thresholds to account for face detection variance
        # OpenCV Haar cascade detects face box, not full head
        min_face_height_percent=50,
        max_face_height_percent=85,
        min_eye_position_percent=40,
        max_eye_position_percent=75,
        min_head_top_margin_percent=2.0,
        max_head_top_margin_percent=25.0,
    ),
    OutputFormat.DIGITAL: PhotoSpec(
        name="Digital Upload (630×810px)",
        width_mm=35.0,  # Corresponds to 35x45mm at screen resolution
        height_mm=45.0,
        width_px=630,  # Official MEA requirement for online submission
        height_px=810,
        min_file_size_kb=10,  # Lowered minimum
        max_file_size_kb=250,  # Passport Seva allows up to 250 KB
        dpi=200,  # Lower DPI for digital
        background_color=(255, 255, 255),  # White
        aspect_ratio=630.0 / 810.0,  # ~0.778
        # Same relaxed thresholds as 35x45mm print
        min_face_height_percent=50,
        max_face_height_percent=85,
        min_eye_position_percent=40,
        max_eye_position_percent=75,
        min_head_top_margin_percent=2.0,
        max_head_top_margin_percent=25.0,
    ),
}

# Validation thresholds
FACE_DETECTION_CONFIDENCE = 0.5
MIN_IMAGE_QUALITY = 0.6  # Minimum quality score (0-1)

# Quality thresholds
# Note: Sharpness score of 100 was too strict for many acceptable photos
# Lowered to 50 to reduce false positives while still catching truly blurry images
MIN_SHARPNESS_SCORE = 50  # Laplacian variance (lower = more lenient)
MIN_BRIGHTNESS = 60  # Allow slightly darker photos
MAX_BRIGHTNESS = 230  # Allow slightly brighter photos
IDEAL_BRIGHTNESS = 140  # Passport photos tend to be well-lit

# Head tilt detection
# Eye level difference as percentage of face height
# 5% was too strict, causing false positives
MAX_HEAD_TILT_PERCENT = 8  # Allow up to 8% eye level difference

# Background validation
BACKGROUND_SAMPLE_REGIONS = [
    (0.0, 0.0, 0.15, 0.15),   # Top-left corner
    (0.85, 0.0, 1.0, 0.15),   # Top-right corner
    (0.0, 0.0, 1.0, 0.05),    # Top strip
]
WHITE_THRESHOLD = 225  # Minimum RGB value to be considered "white"
SHADOW_THRESHOLD = 30  # Max difference between background regions
