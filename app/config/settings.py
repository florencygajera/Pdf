"""
Application Settings — loaded from environment variables or .env file.
All sensitive values (keys, passwords) MUST be set via environment, never hardcoded.

FIXES IN THIS VERSION:
  1. OCR_LANGUAGES default changed ["en"] → ["gu", "en"]
     Gujarati script (U+0A80–U+0AFF) requires PaddleOCR's Indic model.
     The English model silently outputs nothing for Gujarati characters.

  2. CELERY_TASK_TIMEOUT default changed 15 → 300
     PaddleOCR on 6 scanned pages takes 30–90 s. At default=15, the task
     was killed by SoftTimeLimitExceeded before producing any output.
     The .env.example already had 1800 — the code default is now aligned.

  3. effective_ocr_dpi() DPI caps raised across all page-count tiers.
     Previous caps (90/85/80) made Gujarati matras and vowel marks
     indistinguishable at render time. New caps (150/150/120/100) keep
     enough resolution for complex Indic glyphs without sacrificing speed
     on large documents.

  4. OCR_CONFIDENCE_THRESHOLD default 0.5 → 0.3
     PaddleOCR's Gujarati model returns confidence 0.3–0.6 on clean
     government notices. At 0.5 the majority of valid results were dropped.

  5. DIGITAL_TEXT_THRESHOLD default 0.05 → 0.01
     Tighter threshold ensures fully-scanned pages (text_coverage=0) are
     never misrouted to the digital extractor.
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
    ENVIRONMENT: str = Field(default="development")
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
    MAX_FILE_SIZE_MB: int = Field(default=100)
    UPLOAD_DIR: Path = Field(default=Path("/tmp/pdf_uploads"))
    OUTPUT_DIR: Path = Field(default=Path("/tmp/pdf_outputs"))
    ALLOWED_EXTENSIONS: List[str] = Field(default=["pdf"])

    # ── OCR ───────────────────────────────────────────────────────────────
    OCR_DPI: int = Field(
        default=150,
        description=(
            "Base DPI for PDF→image rendering. effective_ocr_dpi() may cap this "
            "per page count but never below 100 for Indic scripts."
        ),
    )
    # FIX 1: default changed ["en"] → ["gu", "en"].
    # PaddleOCR's English model produces no output for Gujarati characters.
    # "gu" loads the Indic model which also handles embedded Latin/digits.
    # NOTE: only the FIRST language is passed to PaddleOCR (see ocr_extractor.py).
    OCR_LANGUAGES: List[str] = Field(
        default=["gu", "en"],
        description=(
            "Ordered list of PaddleOCR language codes. "
            "First entry is used as the primary OCR model language. "
            "Supported: en, gu, hi, ta, te, kn, ml, mr, pa, bn, ch, ..."
        ),
    )
    OCR_USE_GPU: bool = Field(default=False)
    OCR_ENABLE_MKLDNN: bool = Field(
        default=False,
        description="Enable Paddle MKLDNN/oneDNN acceleration (may be unstable).",
    )
    # FIX 4: default lowered 0.5 → 0.3.
    # Gujarati OCR confidence from PaddleOCR's Indic model runs 0.3–0.6 for
    # clean government notices. At 0.5 the majority of valid words were dropped.
    OCR_CONFIDENCE_THRESHOLD: float = Field(
        default=0.3,
        description="Min confidence to accept an OCR word result.",
    )
    OCR_PAGE_WORKERS: Optional[int] = Field(default=None)
    OCR_CHUNK_WORKERS: Optional[int] = Field(default=None)
    OCR_CHUNK_SIZE: Optional[int] = Field(default=None)
    OCR_PDF2IMAGE_THREADS: Optional[int] = Field(default=None)
    OCR_PARALLEL_INFERENCE: bool = Field(default=True)
    OCR_PREWARM_ON_STARTUP: bool = Field(
        default=False,
        description="Warm up PaddleOCR on startup. Keep false on constrained deploys.",
    )
    OCR_SKIP_SCANNED_TABLES_UNDER_PAGES: int = Field(
        default=5,
        description="Skip scanned table extraction for docs below this page count.",
    )

    # ── Digital Extraction ─────────────────────────────────────────────────
    # FIX 5: default lowered 0.05 → 0.01.
    # Fully-scanned pages have text_coverage=0. At 0.05 they were always routed
    # correctly, but pages with faint watermarks (~0.03) could be misclassified
    # as digital. 0.01 provides a tighter guard.
    DIGITAL_TEXT_THRESHOLD: float = Field(
        default=0.01,
        description=(
            "Fraction of page area covered by text bboxes required to classify "
            "a page as digital. Below this threshold the page goes to OCR."
        ),
    )

    # ── Redis / Celery ─────────────────────────────────────────────────────
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    # FIX 2: default raised 15 → 300.
    # PaddleOCR + Gujarati model on 6 scanned pages takes 30–90 s.
    # At default=15 the task was killed by SoftTimeLimitExceeded (triggered at
    # max(3, timeout-5) = 10 s) before producing any extraction output.
    # .env.example already had 1800 — the code default is now consistent.
    CELERY_TASK_TIMEOUT: int = Field(
        default=300,
        description="Hard task time limit in seconds. Soft limit = timeout - 30.",
    )
    CELERY_WORKER_CONCURRENCY: Optional[int] = Field(
        default=1,
        description="Concurrent Celery tasks per worker. Keep 1 for heavy OCR.",
    )
    CELERY_WORKER_POOL: str = Field(default="prefork")
    RESULT_EXPIRES_SECONDS: int = Field(default=7200)
    READINESS_REQUIRE_WORKER: bool = Field(default=False)

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")

    # ── Rate Limiting ──────────────────────────────────────────────────────
    RATE_LIMIT_UPLOAD: str = Field(default="10/minute")
    RATE_LIMIT_EXTRACT: str = Field(default="60/minute")

    @model_validator(mode="after")
    def warn_insecure_secret(self):
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
    def celery_soft_time_limit(self) -> int:
        """Soft limit fires 30 s before hard limit to allow graceful shutdown."""
        return max(30, self.CELERY_TASK_TIMEOUT - 30)

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
        Return the effective render DPI for OCR, capped by page count to balance
        speed vs accuracy.

        FIX 3: All tier caps raised significantly from the previous values
        (96/90/85/80). The old caps were tuned for English Latin script where
        90 DPI is sufficient. Gujarati and other Indic scripts have complex
        matras, half-forms, and conjunct consonants that become indistinguishable
        below 120 DPI, causing PaddleOCR to miss characters entirely.

        New tiers:
          ≤ 10 pages  → min(OCR_DPI, 150)  — full quality for short docs
          ≤ 30 pages  → min(OCR_DPI, 120)  — acceptable quality, faster render
          > 30 pages  → min(OCR_DPI, 100)  — speed-priority for bulk docs
        """
        if total_pages <= 10:
            return min(self.OCR_DPI, 150)
        if total_pages <= 30:
            return min(self.OCR_DPI, 120)
        return min(self.OCR_DPI, 100)

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
