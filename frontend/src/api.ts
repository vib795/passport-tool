import axios from 'axios';
import type { AnalyzeResponse, OutputFormat, PhotoSpec, ProcessResponse } from './types';

// In production (Docker), API is served from same origin
// In development, use localhost:8000
const API_BASE = import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? '' : 'http://localhost:8000');

const api = axios.create({
  baseURL: API_BASE,
});

export async function getSpecs(): Promise<Record<string, PhotoSpec>> {
  const response = await api.get('/specs');
  return response.data;
}

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function processImage(
  file: File,
  options: {
    outputFormat: OutputFormat;
    removeBackground: boolean;
    autoCrop: boolean;
    cropBox?: { x: number; y: number; width: number; height: number };
    brightness?: number;
    contrast?: number;
  }
): Promise<ProcessResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('output_format', options.outputFormat);
  formData.append('remove_background', String(options.removeBackground));
  formData.append('auto_crop', String(options.autoCrop));

  if (options.cropBox) {
    formData.append('crop_x', String(Math.round(options.cropBox.x)));
    formData.append('crop_y', String(Math.round(options.cropBox.y)));
    formData.append('crop_width', String(Math.round(options.cropBox.width)));
    formData.append('crop_height', String(Math.round(options.cropBox.height)));
  }

  if (options.brightness !== undefined) {
    formData.append('brightness', String(options.brightness));
  }
  if (options.contrast !== undefined) {
    formData.append('contrast', String(options.contrast));
  }

  const response = await api.post('/process', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function downloadImage(
  file: File,
  options: {
    outputFormat: OutputFormat;
    removeBackground: boolean;
    autoCrop: boolean;
    imageFormat: 'jpeg' | 'png';
  }
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('output_format', options.outputFormat);
  formData.append('remove_background', String(options.removeBackground));
  formData.append('auto_crop', String(options.autoCrop));
  formData.append('image_format', options.imageFormat);

  const response = await api.post('/download', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  });
  return response.data;
}
