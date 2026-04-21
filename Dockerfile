## ═══════════════════════════════════════════════════════════
# Multi-stage Dockerfile — PDF Extraction Service (Optimized)
# Stage 1: builder  (compile wheels)
# Stage 2: runtime  (lean production image)
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-guj \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Build wheels
RUN pip install --upgrade pip wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt



# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="florencygajera@gmail.com"
LABEL description="Production-grade hybrid PDF extraction service"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    ghostscript \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgdiplus \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle
RUN pip install --no-cache-dir paddlepaddle==2.6.2

# Install Python dependencies from wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

# App setup
WORKDIR /app
COPY . .

# Non-root user
RUN useradd -m -u 1000 extractor && \
    mkdir -p /tmp/pdf_uploads /tmp/pdf_outputs && \
    chown -R extractor:extractor /app /tmp/pdf_uploads /tmp/pdf_outputs

USER extractor

# Environment
ENV ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_FORMAT=json \
    UPLOAD_DIR=/tmp/pdf_uploads \
    OUTPUT_DIR=/tmp/pdf_outputs

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Start server
# Keep a single web worker: Paddle OCR and PDF parsing are memory-heavy, and
# Render already provisions WEB_CONCURRENCY=1 by default for this service.
CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info", \
    "--access-log"]
