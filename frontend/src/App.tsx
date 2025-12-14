import { useCallback, useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import ReactCrop, {
  centerCrop,
  makeAspectCrop,
  type Crop,
  type PixelCrop,
} from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import { getSpecs, processImage } from './api';
import './App.css';
import type { OutputFormat, PhotoSpec, ProcessResponse } from './types';

type Step = 'upload' | 'crop' | 'result';

function App() {
  const [step, setStep] = useState<Step>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [specs, setSpecs] = useState<Record<string, PhotoSpec>>({});
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('digital');
  const [removeBackground, setRemoveBackground] = useState(true);
  const [crop, setCrop] = useState<Crop>();
  const [completedCrop, setCompletedCrop] = useState<PixelCrop>();
  const [result, setResult] = useState<ProcessResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [imgDimensions, setImgDimensions] = useState({ width: 0, height: 0 });
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    getSpecs().then(setSpecs).catch(console.error);
  }, []);

  const currentSpec = specs[outputFormat];
  const aspectRatio = currentSpec?.aspect_ratio ?? 630 / 810;

  const onImageLoad = useCallback(
    (e: React.SyntheticEvent<HTMLImageElement>) => {
      const { width, height } = e.currentTarget;
      setImgDimensions({ width, height });

      // Create a centered crop with correct aspect ratio
      const cropHeight = Math.min(width / aspectRatio, height) * 0.6;
      const cropWidth = cropHeight * aspectRatio;
      const newCrop = centerCrop(
        makeAspectCrop(
          {
            unit: 'px',
            width: cropWidth,
          },
          aspectRatio,
          width,
          height
        ),
        width,
        height
      );
      setCrop(newCrop);
    },
    [aspectRatio]
  );

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const selectedFile = acceptedFiles[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
    setCrop(undefined);
    setCompletedCrop(undefined);
    setBrightness(100);
    setContrast(100);
    setStep('crop');
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    maxFiles: 1,
  });

  const handleProcess = async () => {
    if (!file || !completedCrop || !imgRef.current) return;

    try {
      setLoading(true);
      setError(null);

      // Calculate the scale between displayed and natural image size
      const scaleX = imgRef.current.naturalWidth / imgRef.current.width;
      const scaleY = imgRef.current.naturalHeight / imgRef.current.height;

      const cropBox = {
        x: Math.round(completedCrop.x * scaleX),
        y: Math.round(completedCrop.y * scaleY),
        width: Math.round(completedCrop.width * scaleX),
        height: Math.round(completedCrop.height * scaleY),
      };

      const processResult = await processImage(file, {
        outputFormat,
        removeBackground,
        autoCrop: false,
        cropBox,
        brightness,
        contrast,
      });

      setResult(processResult);

      if (processResult.success) {
        setStep('result');
      } else {
        setError(processResult.message);
      }
    } catch (err) {
      setError('Failed to process image');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result?.image_base64) return;

    const link = document.createElement('a');
    link.href = `data:image/jpeg;base64,${result.image_base64}`;
    link.download = `passport_photo_${outputFormat}.jpg`;
    link.click();
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setCrop(undefined);
    setCompletedCrop(undefined);
    setBrightness(100);
    setContrast(100);
    setStep('upload');
  };

  const handleBackToCrop = () => {
    setResult(null);
    setStep('crop');
  };

  // Update crop when format changes
  const handleFormatChange = (newFormat: OutputFormat) => {
    setOutputFormat(newFormat);
    const newSpec = specs[newFormat];
    if (newSpec && imgDimensions.width > 0) {
      const newAspect = newSpec.aspect_ratio;
      const cropHeight = Math.min(imgDimensions.width / newAspect, imgDimensions.height) * 0.6;
      const cropWidth = cropHeight * newAspect;
      const newCrop = centerCrop(
        makeAspectCrop(
          { unit: 'px', width: cropWidth },
          newAspect,
          imgDimensions.width,
          imgDimensions.height
        ),
        imgDimensions.width,
        imgDimensions.height
      );
      setCrop(newCrop);
    }
  };

  const imageStyle = {
    filter: `brightness(${brightness}%) contrast(${contrast}%)`,
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Indian Passport Photo Tool</h1>
        <p>Resize and format photos for Indian passport applications via VFS Global</p>
      </header>

      <main className="main">
        {/* Step indicator */}
        <div className="steps">
          <div className={`step ${step === 'upload' ? 'active' : 'completed'}`}>
            <span className="step-number">1</span>
            <span className="step-label">Upload</span>
          </div>
          <div className={`step ${step === 'crop' ? 'active' : step === 'result' ? 'completed' : ''}`}>
            <span className="step-number">2</span>
            <span className="step-label">Crop & Adjust</span>
          </div>
          <div className={`step ${step === 'result' ? 'active' : ''}`}>
            <span className="step-number">3</span>
            <span className="step-label">Download</span>
          </div>
        </div>

        {step === 'upload' && (
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'active' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="dropzone-content">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="48"
                height="48"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p>Drag & drop a photo here, or click to select</p>
              <span className="hint">Supports JPG, PNG, WebP - any size</span>
            </div>
          </div>
        )}

        {step === 'crop' && preview && (
          <div className="crop-workspace">
            <div className="crop-sidebar">
              <div className="sidebar-section">
                <h3>Output Format</h3>
                <div className="radio-group">
                  <label className={outputFormat === 'digital' ? 'selected' : ''}>
                    <input
                      type="radio"
                      value="digital"
                      checked={outputFormat === 'digital'}
                      onChange={() => handleFormatChange('digital')}
                    />
                    <div>
                      <strong>Digital Upload</strong>
                      <small>630×810px (Online submission)</small>
                    </div>
                  </label>
                  <label className={outputFormat === 'print_rect' ? 'selected' : ''}>
                    <input
                      type="radio"
                      value="print_rect"
                      checked={outputFormat === 'print_rect'}
                      onChange={() => handleFormatChange('print_rect')}
                    />
                    <div>
                      <strong>Print 35×45mm</strong>
                      <small>Standard passport size</small>
                    </div>
                  </label>
                  <label className={outputFormat === 'print_square' ? 'selected' : ''}>
                    <input
                      type="radio"
                      value="print_square"
                      checked={outputFormat === 'print_square'}
                      onChange={() => handleFormatChange('print_square')}
                    />
                    <div>
                      <strong>Print 51×51mm</strong>
                      <small>2×2 inch (US style)</small>
                    </div>
                  </label>
                </div>
                {currentSpec && (
                  <div className="spec-info">
                    <span>{currentSpec.width_px}×{currentSpec.height_px}px</span>
                    <span>{currentSpec.dpi} DPI</span>
                    <span>{currentSpec.min_file_size_kb}-{currentSpec.max_file_size_kb}KB</span>
                  </div>
                )}
              </div>

              <div className="sidebar-section">
                <h3>Options</h3>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={removeBackground}
                    onChange={(e) => setRemoveBackground(e.target.checked)}
                  />
                  Replace background with white
                </label>
              </div>

              <div className="sidebar-section">
                <h3>Adjustments</h3>
                <div className="slider-group">
                  <label>
                    Brightness: {brightness}%
                    <input
                      type="range"
                      min="50"
                      max="150"
                      value={brightness}
                      onChange={(e) => setBrightness(Number(e.target.value))}
                    />
                  </label>
                </div>
                <div className="slider-group">
                  <label>
                    Contrast: {contrast}%
                    <input
                      type="range"
                      min="50"
                      max="150"
                      value={contrast}
                      onChange={(e) => setContrast(Number(e.target.value))}
                    />
                  </label>
                </div>
                <button
                  className="btn text-btn"
                  onClick={() => {
                    setBrightness(100);
                    setContrast(100);
                  }}
                >
                  Reset adjustments
                </button>
              </div>

              <div className="sidebar-section guide-section">
                <h3>Positioning Guide</h3>
                <div className="guide-preview">
                  <div className="guide-face">
                    <div className="guide-eye-line"></div>
                    <div className="guide-chin-line"></div>
                  </div>
                </div>
                <ul className="guide-tips">
                  <li>Position head within the oval</li>
                  <li>Eyes should align with the upper line</li>
                  <li>Chin near the lower line</li>
                  <li>Face should fill 70-80% of frame</li>
                </ul>
              </div>

              <div className="button-group">
                <button
                  className="btn primary"
                  onClick={handleProcess}
                  disabled={loading || !completedCrop}
                >
                  {loading ? 'Processing...' : 'Create Photo'}
                </button>
                <button className="btn secondary" onClick={handleReset}>
                  Start Over
                </button>
              </div>
            </div>

            <div className="crop-main">
              <div className="crop-container">
                <div className="crop-instructions">
                  <p>Drag the corners to adjust the crop area. Position your face within the frame.</p>
                </div>
                <div className="crop-wrapper">
                  <ReactCrop
                    crop={crop}
                    onChange={(_, percentCrop) => setCrop(percentCrop)}
                    onComplete={(c) => setCompletedCrop(c)}
                    aspect={aspectRatio}
                    minWidth={100}
                    minHeight={100}
                    className="crop-tool"
                  >
                    <img
                      ref={imgRef}
                      src={preview}
                      alt="Upload"
                      onLoad={onImageLoad}
                      style={imageStyle}
                      className="crop-image"
                    />
                  </ReactCrop>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 'result' && result && (
          <div className="result-workspace">
            <div className="result-container">
              <h2>Your Passport Photo</h2>
              {result.image_base64 && (
                <div className="result-image-wrapper">
                  <img
                    src={`data:image/jpeg;base64,${result.image_base64}`}
                    alt="Processed passport photo"
                    className="result-image"
                  />
                </div>
              )}
              {result.validation && (
                <div className="validation-info">
                  {result.validation.is_valid ? (
                    <p className="success">Photo meets requirements</p>
                  ) : (
                    <p className="warning">Photo processed with warnings</p>
                  )}
                  {result.validation.errors.map((err, i) => (
                    <p key={i} className="error">{err}</p>
                  ))}
                  {result.validation.warnings.map((warn, i) => (
                    <p key={i} className="warning">{warn}</p>
                  ))}
                </div>
              )}
              <div className="result-actions">
                <button className="btn download" onClick={handleDownload}>
                  Download Photo
                </button>
                <button className="btn secondary" onClick={handleBackToCrop}>
                  Adjust Crop
                </button>
                <button className="btn text-btn" onClick={handleReset}>
                  Upload New Photo
                </button>
              </div>
              <div className="print-info">
                <h4>Printing Tips</h4>
                <p>Print at actual size (51×51mm or 2×2 inches) on matte photo paper.</p>
                <p>You need 2 identical photos for your passport application.</p>
              </div>
            </div>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        <section className="requirements">
          <h2>Indian Passport Photo Requirements</h2>
          <div className="requirements-grid">
            <div className="requirement">
              <h4>Size</h4>
              <p>51×51mm (2×2 inches)</p>
              <p>600×600px at 300 DPI</p>
            </div>
            <div className="requirement">
              <h4>Background</h4>
              <p>Plain white or light background</p>
            </div>
            <div className="requirement">
              <h4>Face Coverage</h4>
              <p>Face should cover 70-80% of photo</p>
              <p>Full head from hair to chin visible</p>
            </div>
            <div className="requirement">
              <h4>Expression</h4>
              <p>Neutral expression</p>
              <p>Mouth closed, eyes open</p>
            </div>
            <div className="requirement">
              <h4>Accessories</h4>
              <p>No glasses or sunglasses</p>
              <p>No hats (except religious)</p>
            </div>
            <div className="requirement">
              <h4>Clothing</h4>
              <p>Normal street clothes</p>
              <p>No uniforms</p>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>
          For official requirements, visit{' '}
          <a
            href="https://services.vfsglobal.com/usa/en/ind/apply-passport"
            target="_blank"
            rel="noopener noreferrer"
          >
            VFS Global
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
