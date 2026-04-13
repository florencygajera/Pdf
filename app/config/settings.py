"""
Application Settings — loaded from environment variables or .env file.
All sensitive values (keys, passwords) MUST be set via environment, never hardcoded.

FIXES:
  - RATE_LIMIT_EXTRACT default corrected to "60/minute" (was "30/minute" in
    code but "60/minute" in .env.example — now consistent everywhere)
  - effective_ocr_dpi() floor raised from 96 → 110 for very large docs.
    96 DPI was too aggressive and hurt accuracy on dense government tables.

PERFORMANCE KNOBS (set via .env):
  OCR_DPI                  Base render DPI. Default 120. Raise to 150-200 for
                           very small text or poor scan quality.
  OCR_PAGE_WORKERS         Override per-page OCR thread count (hardware-tuned).
  OCR_CHUNK_WORKERS        Override chunk-level process pool workers.
  OCR_CHUNK_SIZE           Override page chunk size for parallel batches.
  OCR_PDF2IMAGE_THREADS    Override pdf2image thread count.
  OCR_PARALLEL_INFERENCE   Set false to force serial OCR (debug only).
  CELERY_WORKER_CONCURRENCY  Celery tasks per worker. Keep 1 for heavy OCR.
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


@lru_cache(maxsize=1)
def _detect_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


@lru_cache(maxsize=1)
def _detect_memory_gb() -> float:
    try:
        import psutil

        return max(1.0, psutil.virtual_memory().total / (1024**3))
    except Exception:
        if os.name == "nt":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                statex = MEMORYSTATUSEX()
                statex.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(statex)):
                    return max(1.0, statex.ullTotalPhys / (1024**3))
            except Exception:
                pass

        return 8.0


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    ENVIRONMENT: str = Field(default="development", description="deployment env")
    DEBUG: bool = Field(default=False)
    SECRET_KEY: str = Field(default="change-me-in-production")
    API_KEY: Optional[str] = Field(default=None)
    TESTING: bool = Field(default=False)

    # ── Server ─────────────────────────────────────────────────────────────
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    ALLOWED_HOSTS: List[str] = Field(default=["*"])
    CORS_ORIGINS: List[str] = Field(default=["*"])

    # ── File Upload ────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = Field(default=100, description="Max upload in MB")
    UPLOAD_DIR: Path = Field(default=Path("/tmp/pdf_uploads"))
    OUTPUT_DIR: Path = Field(default=Path("/tmp/pdf_outputs"))
    ALLOWED_EXTENSIONS: List[str] = Field(default=["pdf"])

    # ── OCR ───────────────────────────────────────────────────────────────
    OCR_DPI: int = Field(default=120, description="DPI for PDF→image conversion")
    OCR_LANGUAGES: List[str] = Field(
        default=["en"],
        description="PaddleOCR language codes e.g. ['en','hi','ch']",
    )
    OCR_USE_GPU: bool = Field(default=False)
    OCR_ENABLE_MKLDNN: bool = Field(
        default=False,
        description="Enable Paddle MKLDNN/oneDNN acceleration for OCR if the runtime is stable",
    )
    OCR_CONFIDENCE_THRESHOLD: float = Field(
        default=0.6, description="Min confidence to accept OCR result"
    )
    OCR_PAGE_WORKERS: Optional[int] = Field(
        default=None,
        description="Override OCR page worker count; defaults to hardware-aware tuning",
    )
    OCR_CHUNK_WORKERS: Optional[int] = Field(
        default=None,
        description="Override OCR chunk worker count; defaults to hardware-aware tuning",
    )
    OCR_CHUNK_SIZE: Optional[int] = Field(
        default=None,
        description="Override OCR chunk size; defaults to hardware-aware tuning",
    )
    OCR_PDF2IMAGE_THREADS: Optional[int] = Field(
        default=None,
        description="Override pdf2image thread_count; defaults to hardware-aware tuning",
    )
    OCR_PARALLEL_INFERENCE: bool = Field(
        default=True,
        description="Allow OCR inference to run concurrently across a bounded pool",
    )

    # ── Digital Extraction ─────────────────────────────────────────────────
    DIGITAL_TEXT_THRESHOLD: float = Field(
        default=0.05,
        description="Fraction of chars that must be extractable for 'digital' classification",
    )

    # ── Redis / Celery ─────────────────────────────────────────────────────
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    CELERY_TASK_TIMEOUT: int = Field(
        default=1800, description="Seconds before task killed"
    )
    CELERY_WORKER_CONCURRENCY: Optional[int] = Field(
        default=1,
        description="Concurrent Celery tasks per worker process. Keep 1 for heavy OCR jobs.",
    )
    CELERY_WORKER_POOL: str = Field(
        default="prefork",
        description="Celery worker pool type. Use prefork for CPU-bound OCR workloads.",
    )
    RESULT_EXPIRES_SECONDS: int = Field(
        default=3600, description="Seconds before completed output files expire"
    )
    READINESS_REQUIRE_WORKER: bool = Field(
        default=False,
        description="If true, readiness requires a Celery worker heartbeat in addition to Redis",
    )

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json", description="'json' or 'text'")

    # ── Rate Limiting ──────────────────────────────────────────────────────
    RATE_LIMIT_UPLOAD: str = Field(default="10/minute")
    # FIX: was "30/minute" in code but "60/minute" in .env.example — aligned to 60
    RATE_LIMIT_EXTRACT: str = Field(default="60/minute")

    @model_validator(mode="after")
    def warn_insecure_secret(self):
        import os

        env = os.environ.get("ENVIRONMENT", self.ENVIRONMENT)
        if self.SECRET_KEY == "change-me-in-production" and (
            env == "production" or self.ENVIRONMENT.lower() == "production"
        ):
            raise ValueError(
                "SECRET_KEY must be changed from the default before running in production."
            )
        return self

    @field_validator(
        "DEBUG", "TESTING", "OCR_USE_GPU", "OCR_ENABLE_MKLDNN", mode="before"
    )
    @classmethod
    def parse_boolish(cls, v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", ""}:
                return False
        return False

    @field_validator("UPLOAD_DIR", "OUTPUT_DIR", mode="before")
    @classmethod
    def create_dirs(cls, v):
        path = Path(v)

        if os.name == "nt" and str(v).startswith("/"):
            path = Path(tempfile.gettempdir()) / str(v).lstrip("/\\")

        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            fallback = Path(tempfile.gettempdir()) / path.name
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    @property
    def max_file_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def is_testing(self) -> bool:
        return bool(self.TESTING or self.ENVIRONMENT.lower() in {"test", "testing"})

    @property
    def cpu_cores(self) -> int:
        return _detect_cpu_count()

    @property
    def memory_gb(self) -> float:
        return _detect_memory_gb()

    @property
    def effective_ocr_page_workers(self) -> int:
        if self.OCR_PAGE_WORKERS and self.OCR_PAGE_WORKERS > 0:
            return self.OCR_PAGE_WORKERS

        cpu = self.cpu_cores
        memory = self.memory_gb

        if not self.OCR_PARALLEL_INFERENCE:
            return 1
        if self.OCR_USE_GPU:
            return 1 if memory < 24 else 2
        if memory < 8:
            return 1
        if memory < 12:
            return min(2, cpu)
        if memory < 24:
            return min(4, max(2, cpu // 2))
        return min(4, max(3, cpu // 2))

    @property
    def effective_ocr_chunk_workers(self) -> int:
        if self.OCR_CHUNK_WORKERS and self.OCR_CHUNK_WORKERS > 0:
            return self.OCR_CHUNK_WORKERS

        cpu = self.cpu_cores
        memory = self.memory_gb

        if memory < 8:
            return 1
        if memory < 12:
            return min(2, max(1, cpu // 4 or 1))
        if memory < 20:
            return min(4, max(2, cpu // 2 or 1))
        if memory < 32:
            return min(5, max(3, cpu // 2 or 1))
        return min(6, max(4, cpu - 1))

    @property
    def effective_celery_worker_concurrency(self) -> int:
        if self.CELERY_WORKER_CONCURRENCY and self.CELERY_WORKER_CONCURRENCY > 0:
            return self.CELERY_WORKER_CONCURRENCY
        return 1

    @property
    def effective_ocr_pdf2image_threads(self) -> int:
        if self.OCR_PDF2IMAGE_THREADS and self.OCR_PDF2IMAGE_THREADS > 0:
            return self.OCR_PDF2IMAGE_THREADS
        if self.memory_gb >= 16 and self.cpu_cores >= 8:
            return 4
        if self.cpu_cores >= 4:
            return 2
        return 1

    def effective_ocr_dpi(self, total_pages: int) -> int:
        """
        Return effective render DPI, scaling down for large documents to
        reduce rendering time while preserving accuracy.

        FIX: Floor raised from 96 → 110.
          96 DPI on dense government tables (>60 pages) was too aggressive —
          character recognition dropped noticeably on small-font text.
          110 DPI is a safer lower bound; still ~35% faster than 150 DPI.

        Thresholds (pages → max DPI):
          ≤ 4 pages   → min(OCR_DPI, 120)
          ≤ 20 pages  → min(OCR_DPI, 115)
          ≤ 60 pages  → min(OCR_DPI, 112)
          > 60 pages  → min(OCR_DPI, 110)   ← was 96, raised to 110
        """
        if total_pages <= 4:
            return min(self.OCR_DPI, 120)
        if total_pages <= 20:
            return min(self.OCR_DPI, 115)
        if total_pages <= 60:
            return min(self.OCR_DPI, 112)
        return min(self.OCR_DPI, 110)  # FIX: was 96

    def effective_ocr_chunk_size(self, total_pages: int) -> int:
        if self.OCR_CHUNK_SIZE and self.OCR_CHUNK_SIZE > 0:
            return self.OCR_CHUNK_SIZE

        total_pages = max(1, total_pages)
        workers = max(1, self.effective_ocr_page_workers)
        memory = self.memory_gb

        if total_pages <= workers * 2:
            return total_pages

        if memory < 12:
            base = 5
        elif memory < 24:
            base = 8
        else:
            base = 10

        target = max(4, total_pages // max(workers * 2, 1))
        return max(4, min(10, max(base, target)))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
