# Indian Passport Photo Tool

A free, open-source tool for resizing and formatting photos for Indian passport applications through VFS Global and Passport Seva.

## Features

- **Multiple output formats:**
  - 51x51mm (2x2 inch) - Standard print format
  - 35x45mm (3.5x4.5cm) - Alternative print format
  - 630x810px - Digital upload format for online applications

- **Interactive crop tool** with face positioning guides (oval, eye line, chin line)
- **Automatic background removal** and white background replacement
- **Brightness and contrast adjustment**
- **Face detection** for auto-cropping suggestions
- **Photo validation** based on Indian passport requirements (see below)

## Privacy

This tool requires a backend server for image processing (background removal, face detection). Here's how your photos are handled:

- **Self-hosted (recommended)**: When you run this locally or on your own server, photos never leave your machine/infrastructure. They are processed in memory and never written to disk.

- **Public hosting**: If someone hosts this publicly, photos are uploaded to their server for processing. While the code doesn't store images to disk (they're processed in memory only), you are trusting the server operator. For sensitive documents like passport photos, we recommend self-hosting.

## Tech Stack

**Backend:**
- Python 3.12+
- FastAPI
- OpenCV (face detection)
- rembg (background removal)
- Pillow (image processing)

**Frontend:**
- React 18
- TypeScript
- Vite
- react-image-crop
- react-dropzone

## Prerequisites

- Python 3.12 or higher
- Node.js 18 or higher
- [uv](https://github.com/astral-sh/uv) package manager (for Python)

## Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/passport-tool.git
cd passport-tool
```

### 2. Backend Setup

```bash
# Install Python dependencies using uv
uv sync

# Start the backend server
uv run uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Docker Setup

Build and run using Docker Compose:

```bash
docker-compose up --build
```

Or build the Docker image directly:

```bash
docker build -t passport-tool .
docker run -p 8000:8000 passport-tool
```

Or run:
```bash
./scripts/dev.sh
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/specs` | Get all photo format specifications |
| POST | `/analyze` | Analyze image for face detection and quality |
| POST | `/process` | Process image with crop, brightness, contrast |
| POST | `/download` | Download processed image as file |

## Photo Specifications

Based on MEA Passport Seva and VFS Global guidelines (2024).

### Print Square (51x51mm / 2x2 inch)
- Dimensions: 600x600 pixels at 300 DPI
- Face height: 40-75% of photo
- File size: 10-1000 KB

### Print Standard (35x45mm / 3.5x4.5cm)
- Dimensions: 413x531 pixels at 300 DPI
- Face height: 50-85% of photo
- File size: 10-1000 KB

### Digital Upload (for online applications)
- Dimensions: 630x810 pixels (exact)
- Face height: 50-85% of photo
- File size: 10-250 KB (Passport Seva limit)

## Validation Checks

The tool validates your photo against Indian passport requirements:

| Check | Description |
|-------|-------------|
| Face detection | Ensures a clear frontal face is detected |
| Face size | Verifies face occupies correct percentage of frame |
| Eye position | Checks eyes are at proper height in the photo |
| Head tilt | Detects if head is tilted (should be straight) |
| Head margin | Ensures proper space above head |
| Face centering | Checks face is horizontally centered |
| Background color | Validates white/off-white background |
| Shadow detection | Warns about uneven lighting or shadows |
| Image sharpness | Detects blurry photos |
| Brightness | Checks for proper exposure (not too dark/bright) |
| Contrast | Ensures face features are clearly visible |
| Resolution | Verifies minimum image resolution |

**Note:** This tool provides automated validation to help catch common issues, but it cannot verify all requirements (e.g., neutral expression, appropriate attire). Always review the final photo yourself before submission.

## Usage

1. **Upload** - Drag and drop or click to select your photo
2. **Crop** - Use the interactive crop tool to position your face within the guides
   - Blue oval: Face placement guide
   - Yellow lines: Eye and chin positioning guides
3. **Adjust** - Fine-tune brightness and contrast if needed
4. **Process** - Click "Process Photo" to apply background removal and formatting
5. **Download** - Save your passport-ready photo

## Project Structure

```
passport-tool/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and routes
│   ├── config.py            # Photo specifications and settings
│   └── image_processor.py   # Image processing logic
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main React component
│   │   ├── App.css          # Styles
│   │   ├── api.ts           # API client
│   │   └── types.ts         # TypeScript types
│   ├── package.json
│   └── vite.config.ts
├── scripts/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Disclaimer

This is a personal project and is **not affiliated** with VFS Global, the Ministry of External Affairs (MEA), or any government agency.

**Please verify your photos meet current official requirements before submission.** The creator is not responsible for rejected applications. Always double-check your work against the latest guidelines at:

- [Passport Seva](https://www.passportindia.gov.in/)
- [VFS Global](https://www.vfsglobal.com/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.
