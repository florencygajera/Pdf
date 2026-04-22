"""
Pydantic models for all API request/response contracts.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    filename: str
    size_bytes: int
    message: str = "File uploaded. Use job_id to poll /extract/{job_id}."


# ── Extraction Result ─────────────────────────────────────────────────────────


class TableData(BaseModel):
    page: int
    table_index: int
    headers: List[str]
    rows: List[List[Any]]
    extraction_method: str = Field(
        ..., description="'pdfplumber', 'ocr-grid', 'unknown'"
    )


class PageResult(BaseModel):
    page_number: int
    text: str
    tables: List[TableData] = []
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Per-page OCR/extraction confidence"
    )
    warnings: List[str] = []


class ExtractionMetadata(BaseModel):
    pages: int
    pdf_type: str = Field(..., description="'digital', 'scanned', or 'mixed'")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time_seconds: float
    languages_detected: List[str] = []
    ocr_engine: Optional[str] = None
    warnings: List[str] = []
    warnings_truncated: bool = False
    total_warning_count: int = 0


class ExtractionResult(BaseModel):
    job_id: str
    status: str = Field(..., description="'done', 'processing', 'failed'")
    text: str = Field(default="", description="Full merged clean text")
    full_text: str = Field(default="", description="Alias for full merged clean text")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall extraction confidence"
    )
    language: str = Field(default="unknown", description="Detected document language")
    tables: List[TableData] = []
    pages: List[PageResult] = []
    metadata: ExtractionMetadata


# ── Job Status ────────────────────────────────────────────────────────────────


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress_percent: float = 0.0
    message: str = ""


# ── Error ─────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    job_id: Optional[str] = None
