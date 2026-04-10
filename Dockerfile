# ═══════════════════════════════════════════════════════════
# Multi-stage Dockerfile — PDF Extraction Service
# Stage 1: builder  (install deps, compile wheels)
# Stage 2: runtime  (lean final image, no build tools)
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps needed to compile Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Build wheels for all packages (faster runtime install)
RUN pip install --upgrade pip wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="florencygajera@gmail.com"
LABEL description="Production-grade hybrid PDF extraction service"

# ── System dependencies (runtime only) ───────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # pdf2image (poppler)
    poppler-utils \
    # Ghostscript (legacy PDF compatibility / optional tooling)
    ghostscript \
    # OpenCV
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    # PaddleOCR font rendering
    libgdiplus \
    # Networking
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── PaddlePaddle CPU (install separately for size control) ───
RUN pip install --no-cache-dir paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html || \
    pip install --no-cache-dir paddlepaddle

# ── Install pre-built wheels ──────────────────────────────────
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

# ── App code ─────────────────────────────────────────────────
WORKDIR /app
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 extractor && \
    mkdir -p /tmp/pdf_uploads /tmp/pdf_outputs && \
    chown -R extractor:extractor /app /tmp/pdf_uploads /tmp/pdf_outputs

USER extractor

# ── Environment defaults (override via docker run -e or .env) ─
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_FORMAT=json \
    UPLOAD_DIR=/tmp/pdf_uploads \
    OUTPUT_DIR=/tmp/pdf_outputs

EXPOSE 8000

# ── Health check ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# ── Default command: FastAPI via uvicorn ───────────────────────
CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--log-level", "info", \
    "--access-log"]
