# Multi-stage build for passport photo tool

# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend with frontend static files
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY backend/ ./backend/

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./static

# Create a simple server that serves both API and static files
RUN echo 'from fastapi.staticfiles import StaticFiles\n\
from fastapi.responses import FileResponse\n\
from backend.main import app\n\
import os\n\
\n\
# Mount static files\n\
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")\n\
\n\
@app.get("/")\n\
async def serve_index():\n\
    return FileResponse("static/index.html")\n\
\n\
@app.get("/{path:path}")\n\
async def serve_static(path: str):\n\
    file_path = f"static/{path}"\n\
    if os.path.exists(file_path):\n\
        return FileResponse(file_path)\n\
    return FileResponse("static/index.html")\n\
' > serve.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
