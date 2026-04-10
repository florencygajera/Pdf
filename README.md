# 📄 Hybrid PDF Extraction System

Production-grade extraction pipeline for government documents. Handles **digital PDFs**, **scanned PDFs**, and **mixed documents** with near-zero error rates.

---

## 🏗️ Architecture

```
Upload → PDF Type Detection → [Digital Engine | OCR Engine]
       → Table Extraction → Layout Reconstruction
       → Noise Removal → Validation → JSON Output
```

| Layer | Technology |
|---|---|
| Digital Extraction | PyMuPDF (fitz) |
| OCR | PaddleOCR (CPU/GPU) |
| Image Preprocessing | OpenCV |
| PDF→Image | pdf2image + Poppler |
| Table Extraction (digital) | pdfplumber |
| Table Extraction (scanned) | OpenCV grid detection |
| Layout Detection | Coordinate heuristics + optional LayoutParser |
| API | FastAPI + Async |
| Task Queue | Celery + Redis |
| Containerization | Docker + docker-compose |

---

## 📁 Project Structure

```
pdf_extractor/
├── app/
│   ├── main.py                     # FastAPI app factory
│   ├── config/
│   │   ├── settings.py             # Pydantic-settings (env vars)
│   │   └── constants.py            # System-wide constants
│   ├── api/routes/
│   │   ├── health.py               # /healthz, /readyz
│   │   ├── upload.py               # POST /api/v1/upload
│   │   └── extract.py              # GET /api/v1/extract/{job_id}
│   ├── services/
│   │   ├── pdf_detector.py         # Digital vs Scanned classification
│   │   ├── digital_extractor.py    # PyMuPDF text extraction
│   │   ├── ocr_extractor.py        # PaddleOCR pipeline
│   │   ├── layout_engine.py        # Multi-column reading order
│   │   ├── table_extractor.py      # pdfplumber + OCR grid tables
│   │   ├── noise_cleaner.py        # Dedup, regex, watermark removal
│   │   └── validator.py            # Confidence scoring, QA
│   ├── pipelines/
│   │   └── extraction_pipeline.py  # Master orchestrator
│   ├── utils/
│   │   ├── file_handler.py         # Streaming upload, temp file lifecycle
│   │   ├── image_preprocessing.py  # OpenCV pipeline (deskew, denoise)
│   │   ├── sorting.py              # Reading-order sort algorithms
│   │   └── logger.py              # Structured JSON/text logger
│   ├── models/
│   │   └── response_model.py       # Pydantic I/O contracts
│   └── workers/
│       └── celery_worker.py        # Async extraction task
├── tests/
│   ├── conftest.py
│   └── test_all.py                 # 20+ unit + integration tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Option A — Docker Compose (recommended)

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env as needed

# 2. Start all services (API + Celery worker + Redis)
docker compose up --build

# 3. Upload a PDF
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@/path/to/document.pdf"
# → {"job_id": "abc-123", ...}

# 4. Poll for result
curl http://localhost:8000/api/v1/extract/abc-123
```

### Option B — Local Development

```bash
# Prerequisites: Python 3.11+, Redis, Poppler

# 1. Install system deps (Ubuntu/Debian)
sudo apt-get install poppler-utils

# 2. Install Python packages
pip install paddlepaddle          # CPU version
pip install -r requirements.txt
# Optional: only if you need ML-based layout analysis
# pip install -r requirements-optional.txt

# 3. Start Redis
redis-server &

# 4. Start Celery worker
celery -A app.workers.celery_worker worker --loglevel=info &

# 5. Start API
uvicorn app.main:app --reload --port 8000
```

---

## 🔌 API Reference

### `POST /api/v1/upload`
Upload a PDF for extraction.

**Request:** `multipart/form-data` with `file` field  
**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "gazette.pdf",
  "size_bytes": 5242880,
  "message": "File uploaded. Use job_id to poll /extract/{job_id}."
}
```

### `GET /api/v1/extract/{job_id}`
Poll for result. Returns **202** while processing, **200** when done.

**200 Response:**
```json
{
  "job_id": "...",
  "status": "done",
  "text": "Full extracted clean text...",
  "tables": [
    {
      "page": 3,
      "table_index": 0,
      "headers": ["Name", "Rank", "Unit"],
      "rows": [["Ramesh Kumar", "Havildar", "5 Para SF"]],
      "extraction_method": "pdfplumber"
    }
  ],
  "pages": [...],
  "metadata": {
    "pages": 12,
    "pdf_type": "mixed",
    "confidence_score": 0.87,
    "processing_time_seconds": 34.2,
    "languages_detected": ["en", "hi"],
    "ocr_engine": "PaddleOCR"
  }
}
```

### `GET /api/v1/extract/{job_id}/status` — lightweight progress check
### `GET /api/v1/extract/{job_id}/text` — text only
### `GET /api/v1/extract/{job_id}/tables` — tables only  
### `DELETE /api/v1/extract/{job_id}` — cancel and cleanup
### `GET /healthz` — liveness probe
### `GET /readyz` — readiness probe (checks Redis)

---

## ⚙️ Configuration

All settings are via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MAX_FILE_SIZE_MB` | `100` | Max upload size |
| `OCR_DPI` | `150` | PDF→image resolution |
| `OCR_LANGUAGES` | `["en"]` | PaddleOCR languages |
| `OCR_USE_GPU` | `false` | Enable GPU OCR |
| `OCR_CONFIDENCE_THRESHOLD` | `0.6` | Min OCR confidence |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker |
| `CELERY_TASK_TIMEOUT` | `600` | Seconds before task kill |
| `LOG_FORMAT` | `json` | `json` or `text` |

---

## 🧪 Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Specific test class
pytest tests/test_all.py::TestNoiseCleaner -v

# Just API tests
pytest tests/test_all.py::TestAPIRoutes -v
```

---

## 🌐 Multi-Language Support

Set `OCR_LANGUAGES` in your `.env`:
```
OCR_LANGUAGES=["en","hi"]    # English + Hindi
OCR_LANGUAGES=["en","ch"]    # English + Chinese
```

PaddleOCR supports 80+ languages. See the [PaddleOCR docs](https://github.com/PaddlePaddle/PaddleOCR) for language codes.

---

## 📊 Monitoring (Flower)

```bash
# Start with monitoring profile
docker compose --profile monitoring up

# Access Flower dashboard
open http://localhost:5555
```

---

## 🔒 Production Checklist

- [ ] Set strong `SECRET_KEY` in `.env`
- [ ] Set `ENVIRONMENT=production` (disables `/docs`)
- [ ] Configure `ALLOWED_HOSTS` and `CORS_ORIGINS`
- [ ] Set `LOG_FORMAT=json` for log aggregation
- [ ] Mount persistent volumes for `uploads/` and `outputs/`
- [ ] Set resource limits for Celery workers
- [ ] Configure Redis persistence (AOF/RDB)
- [ ] Set up log rotation

---

## 🐛 Troubleshooting

**`pdf2image` fails:** Install Poppler: `sudo apt-get install poppler-utils`  
**Digital tables empty:** The digital table path uses `pdfplumber`; ensure the PDF has selectable text.  
**PaddleOCR import error:** `pip install paddlepaddle paddleocr`  
**Memory crash on large files:** Reduce `CHUNK_SIZE_PAGES` in `constants.py`  
**Low OCR accuracy:** Increase `OCR_DPI` to 400, ensure good scan quality
