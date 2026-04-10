"""
Application Settings — loaded from environment variables or .env file.
All sensitive values (keys, passwords) MUST be set via environment, never hardcoded.
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


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
    OCR_DPI: int = Field(default=300, description="DPI for PDF→image conversion")
    OCR_LANGUAGES: List[str] = Field(
        default=["en"],
        description="PaddleOCR language codes e.g. ['en','hi','ch']",
    )
    OCR_USE_GPU: bool = Field(default=False)
    OCR_CONFIDENCE_THRESHOLD: float = Field(
        default=0.6, description="Min confidence to accept OCR result"
    )

    # ── Digital Extraction ─────────────────────────────────────────────────
    DIGITAL_TEXT_THRESHOLD: float = Field(
        default=0.05,
        description="Fraction of chars that must be extractable for 'digital' classification",
    )

    # ── Redis / Celery ─────────────────────────────────────────────────────
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    CELERY_TASK_TIMEOUT: int = Field(
        default=600, description="Seconds before task killed"
    )

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json", description="'json' or 'text'")

    # ── Rate Limiting ──────────────────────────────────────────────────────
    RATE_LIMIT_UPLOAD: str = Field(default="10/minute")
    RATE_LIMIT_EXTRACT: str = Field(default="30/minute")

    @field_validator("DEBUG", "TESTING", "OCR_USE_GPU", mode="before")
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
            if normalized in {"1", "true", "yes", "y", "on", "debug", "dev", "development"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", "release", "prod", "production", ""}:
                return False
        return False

    @field_validator("UPLOAD_DIR", "OUTPUT_DIR", mode="before")
    @classmethod
    def create_dirs(cls, v):
        path = Path(v)

        # Windows tests often inject POSIX-style `/tmp/...` paths. Normalize them
        # to the local temp directory so the app can boot in restricted sandboxes.
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

@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
