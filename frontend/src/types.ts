export interface PhotoSpec {
  name: string;
  width_mm: number;
  height_mm: number;
  width_px: number;
  height_px: number;
  min_file_size_kb: number;
  max_file_size_kb: number;
  dpi: number;
  aspect_ratio: number;
}

export interface FaceDetection {
  detected: boolean;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Validation {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface ProcessResponse {
  success: boolean;
  message: string;
  image_base64?: string;
  face_detection?: FaceDetection;
  validation?: Validation;
}

export interface AnalyzeResponse {
  success: boolean;
  message: string;
  width: number;
  height: number;
  face_detection?: FaceDetection;
  quality_score: number;
  quality_issues: string[];
}

export type OutputFormat = 'print_square' | 'print_rect' | 'digital';
