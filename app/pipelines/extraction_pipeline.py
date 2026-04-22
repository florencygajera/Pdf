"""
Extraction Pipeline — Master Orchestrator

Performance-focused version:
  - reads the PDF once into memory
  - classifies from in-memory bytes
  - runs digital and OCR branches concurrently
  - FIX: uses extract_tables_digital_batch — ONE pdfplumber open for all pages
  - FIX: reads stitched pages from "page_results" key (validator fix)
  - preserves stable output ordering and existing API contracts

FIXES IN THIS VERSION:
  FIX 1 — effective_ocr_dpi() now called with doc_classification.total_pages
           (was called with len(scanned_page_nums) — a count that could be 1
           even for a 50-page mixed PDF, causing DPI to never be scaled down).
  FIX 2 — _resolve_scanned_ocr_language() reads settings.OCR_ENGINE and
           settings.OCR_LANGUAGE correctly; added "gu" as explicit Gujarati
           trigger so .env OCR_LANGUAGE=gu routes to Tesseract without needing
           the full word "gujarati".
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from app.config.constants import (
    PDF_TYPE_DIGITAL,
    PDF_TYPE_SCANNED,
    STATE_DONE,
)
from app.config.settings import settings
from app.models.response_model import (
    ExtractionMetadata,
    ExtractionResult,
    PageResult,
    TableData,
)
from app.services.digital_extractor import extract_digital_pdf
from app.services.gujarati_ocr import extract_gujarati_pdf, tesseract_available
from app.services.noise_cleaner import clean_pages, clean_text_block
from app.services.ocr_extractor import extract_ocr_pdf, extract_ocr_pdf_from_bytes
from app.services.pdf_detector import DocumentClassification, detect_pdf_type_from_bytes

# FIX: import the new batch function instead of per-page extract_tables_digital
from app.services.table_extractor import extract_tables_digital_batch
from app.services.validator import validate_extraction_result
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_scanned_ocr_language() -> str:
    """
    Resolve the OCR language used for scanned pages.

    FIX 2: "gu" is now treated as a Gujarati trigger (alongside "guj"/"gujarati").
    This matches the .env convention OCR_LANGUAGE=gu used in docker-compose.yml.

    Routing:
      - Gujarati → Tesseract only (via gujarati_ocr.py)
      - English  → PaddleOCR only (via ocr_extractor.py)
    """
    engine = (settings.OCR_ENGINE or "hybrid").strip().lower()
    configured = (getattr(settings, "OCR_LANGUAGE", None) or "").strip().lower()

    if engine == "tesseract":
        return "gujarati"
    if engine == "paddle":
        return "english"

    # FIX 2: treat "gu" as Gujarati (was only checking "guj" and "gujarati")
    if configured in {"gu", "guj", "gujarati"}:
        return "gujarati"

    # Also check OCR_LANGUAGES[0] as fallback
    if settings.OCR_LANGUAGES:
        first = settings.OCR_LANGUAGES[0].strip().lower()
        if first in {"gu", "guj", "gujarati"}:
            return "gujarati"

    return "english"


def _display_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in {"gu", "guj", "gujarati"}:
        return "gujarati"
    if normalized in {"en", "eng", "english"}:
        return "english"
    return normalized or "unknown"


def _prefer_gujarati_tesseract_ocr() -> bool:
    return _resolve_scanned_ocr_language() == "gujarati" and tesseract_available()


def _build_page_result(
    page_number: int,
    text: str,
    tables: List[Dict],
    confidence: float,
    warnings: List[str],
) -> PageResult:
    """Construct a PageResult model from raw data."""
    table_models = [
        TableData(
            page=t.get("page", page_number),
            table_index=t.get("table_index", 0),
            headers=t.get("headers", []),
            rows=t.get("rows", []),
            extraction_method=t.get("extraction_method", "unknown"),
        )
        for t in tables
    ]
    return PageResult(
        page_number=page_number,
        text=text,
        tables=table_models,
        confidence=confidence,
        warnings=warnings,
    )


def _run_with_progress_lock(
    progress_callback: Optional[Callable],
    lock: Lock,
    step: int,
    total: int,
    stage: str,
) -> None:
    if not progress_callback:
        return
    with lock:
        progress_callback(step, total, stage)


def _process_digital_pages(
    pdf_path: Path,
    doc_classification: DocumentClassification,
    progress_callback: Optional[Callable] = None,
    pdf_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    """
    Process all digital pages in a single extraction pass, then batch extract tables.

    FIX: Uses extract_tables_digital_batch — opens pdfplumber exactly ONCE for
    all pages that have text, instead of once-per-page (which was O(n) file opens).
    Only pages with actual text content go through table extraction.
    """
    digital_page_nums = [
        p.page_number
        for p in doc_classification.pages
        if p.pdf_type == PDF_TYPE_DIGITAL
    ]

    if not digital_page_nums:
        return []

    pdf_bytes = pdf_bytes or pdf_path.read_bytes()
    progress_lock = Lock()

    logger.info(
        "Processing %s digital pages in a single pass",
        len(digital_page_nums),
    )

    page_data = extract_digital_pdf(
        pdf_path, page_numbers=digital_page_nums, pdf_bytes=pdf_bytes
    )

    # FIX: only extract tables for pages that have text (skip blank pages)
    pages_with_text = [
        page_info["page_number"]
        for page_info in page_data
        if page_info.get("text", "").strip()
    ]

    if pages_with_text:
        # FIX: single batch open — was previously one pdfplumber.open() per page
        table_map = extract_tables_digital_batch(
            pdf_path, pages_with_text, pdf_bytes=pdf_bytes
        )
        for page_info in page_data:
            pnum = page_info["page_number"]
            page_info["tables"] = table_map.get(pnum, [])
            page_info["confidence"] = 0.95
    else:
        for page_info in page_data:
            page_info.setdefault("tables", [])
            page_info["confidence"] = page_info.get("confidence", 0.95)

    _run_with_progress_lock(progress_callback, progress_lock, 1, 1, "digital")
    return sorted(page_data, key=lambda p: p.get("page_number", 0))


def _process_scanned_pages(
    pdf_path: Path,
    doc_classification: DocumentClassification,
    progress_callback: Optional[Callable] = None,
    pdf_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    """
    Process scanned pages through a shared OCR executor.

    FIX 1: DPI is now computed using doc_classification.total_pages (the full
    document page count) rather than len(scanned_page_nums). This ensures that
    a mixed 50-page PDF where only 5 pages are scanned still gets the capped DPI
    (100 for >30 pages) instead of the short-doc DPI (150 for ≤10 pages).
    """
    scanned_page_nums = [
        p.page_number
        for p in doc_classification.pages
        if p.pdf_type == PDF_TYPE_SCANNED
    ]

    if not scanned_page_nums:
        return []

    progress_lock = Lock()
    logger.info(
        "Processing %s scanned pages via shared OCR executor",
        len(scanned_page_nums),
    )

    # FIX 1: use total_pages not len(scanned_page_nums) for DPI scaling
    dpi = settings.effective_ocr_dpi(doc_classification.total_pages)

    if _prefer_gujarati_tesseract_ocr():
        logger.info(
            "Using Gujarati Tesseract OCR | pages=%s | dpi=%s",
            len(scanned_page_nums),
            dpi,
        )
        results = extract_gujarati_pdf(
            pdf_path,
            page_numbers=scanned_page_nums,
            dpi=dpi,
            pdf_bytes=pdf_bytes,
        )
    else:
        if pdf_bytes:
            results = extract_ocr_pdf_from_bytes(
                pdf_bytes,
                page_numbers=scanned_page_nums,
                dpi=dpi,
                language=_resolve_scanned_ocr_language(),
            )
        else:
            results = extract_ocr_pdf(
                pdf_path,
                page_numbers=scanned_page_nums,
                dpi=dpi,
                language=_resolve_scanned_ocr_language(),
            )

    _run_with_progress_lock(progress_callback, progress_lock, 1, 1, "ocr")
    return sorted(results, key=lambda p: p.get("page_number", 0))


def run_extraction_pipeline(
    pdf_path: Path,
    job_id: str,
    progress_callback: Optional[Callable] = None,
    pdf_bytes: Optional[bytes] = None,
) -> ExtractionResult:
    """
    Master pipeline: detect → extract → clean → validate → output.
    """
    start_time = time.time()
    warnings: List[str] = []

    # Read once. Reuse bytes across detection, digital extraction, and OCR.
    if pdf_bytes is None:
        try:
            pdf_bytes = pdf_path.read_bytes()
        except Exception as exc:
            raise ValueError(f"Failed to read PDF: {exc}") from exc

    logger.info("[%s] Step 1: PDF type detection", job_id)
    try:
        doc_classification = detect_pdf_type_from_bytes(
            pdf_bytes, file_name=pdf_path.name
        )
    except ValueError as exc:
        raise ValueError(f"PDF validation failed: {exc}") from exc

    logger.info("[%s] Step 2: Extracting content (digital + OCR concurrently)", job_id)

    digital_results: List[Dict] = []
    scanned_results: List[Dict] = []

    has_digital = any(p.pdf_type == PDF_TYPE_DIGITAL for p in doc_classification.pages)
    has_scanned = any(p.pdf_type == PDF_TYPE_SCANNED for p in doc_classification.pages)

    if has_digital and has_scanned:
        with ThreadPoolExecutor(max_workers=2) as branch_pool:
            digital_future = branch_pool.submit(
                _process_digital_pages,
                pdf_path,
                doc_classification,
                None,
                pdf_bytes,
            )
            scanned_future = branch_pool.submit(
                _process_scanned_pages,
                pdf_path,
                doc_classification,
                None,
                pdf_bytes,
            )
            done_count = 0
            for fut in (digital_future, scanned_future):
                result = fut.result()
                done_count += 1
                if progress_callback:
                    progress_callback(done_count, 2, "extraction")
                if fut is digital_future:
                    digital_results = result
                else:
                    scanned_results = result
    elif has_digital:
        digital_results = _process_digital_pages(
            pdf_path, doc_classification, progress_callback, pdf_bytes
        )
    else:
        scanned_results = _process_scanned_pages(
            pdf_path, doc_classification, progress_callback, pdf_bytes
        )

    all_page_results: List[Dict] = sorted(
        digital_results + scanned_results,
        key=lambda p: p.get("page_number", 0),
    )

    logger.info("[%s] Step 3: Noise removal", job_id)
    page_texts = [page.get("text", "") for page in all_page_results]
    cleaned_texts = clean_pages(page_texts)
    for idx, cleaned in enumerate(cleaned_texts):
        all_page_results[idx]["text"] = cleaned

    all_tables: List[Dict] = []
    for page in all_page_results:
        all_tables.extend(page.get("tables", []))

    logger.info("[%s] Step 4: Validation pass", job_id)
    validation_report = validate_extraction_result(all_page_results, all_tables)

    # FIX: use "page_results" key (was "stitched_pages" which wasn't returned by validator)
    stitched_pages = validation_report.get("page_results") or validation_report.get(
        "stitched_pages"
    )
    if stitched_pages:
        for idx, page in enumerate(stitched_pages):
            if idx < len(all_page_results):
                all_page_results[idx]["text"] = page.get(
                    "text", all_page_results[idx].get("text", "")
                )
                all_page_results[idx]["warnings"] = page.get(
                    "warnings", all_page_results[idx].get("warnings", [])
                )
    warnings.extend(
        f"Page {pn}: {w}"
        for pn, ws in validation_report.get("page_warnings", {}).items()
        for w in ws
    )

    logger.info("[%s] Step 5: Assembling output", job_id)

    full_text = "\n\n---\n\n".join(
        f"[Page {p.get('page_number', i + 1)}]\n{p.get('text', '').strip()}"
        for i, p in enumerate(all_page_results)
        if p.get("text", "").strip()
    )
    full_text = clean_text_block(full_text)

    page_models: List[PageResult] = [
        _build_page_result(
            page_number=p.get("page_number", i + 1),
            text=p.get("text", ""),
            tables=p.get("tables", []),
            confidence=p.get("confidence", 0.0),
            warnings=p.get("warnings", []),
        )
        for i, p in enumerate(all_page_results)
    ]

    table_models: List[TableData] = [
        TableData(
            page=t.get("page", 0),
            table_index=t.get("table_index", 0),
            headers=t.get("headers", []),
            rows=t.get("rows", []),
            extraction_method=t.get("extraction_method", "unknown"),
        )
        for t in all_tables
    ]

    processing_time = round(time.time() - start_time, 3)
    total_warning_count = len(warnings)
    warnings_truncated = total_warning_count > 50

    metadata = ExtractionMetadata(
        pages=doc_classification.total_pages,
        pdf_type=doc_classification.overall_type,
        confidence_score=validation_report["overall_confidence"],
        processing_time_seconds=processing_time,
        languages_detected=validation_report.get("languages", []),
        ocr_engine=(
            "Tesseract"
            if doc_classification.scanned_page_count > 0
            and _resolve_scanned_ocr_language() == "gujarati"
            else "Paddle"
            if doc_classification.scanned_page_count > 0
            else None
        ),
        warnings=warnings[:50],
        warnings_truncated=warnings_truncated,
        total_warning_count=total_warning_count,
    )

    result = ExtractionResult(
        job_id=job_id,
        status=STATE_DONE,
        text=full_text,
        full_text=full_text,
        confidence=metadata.confidence_score,
        language=(
            "gujarati"
            if doc_classification.scanned_page_count > 0
            and _resolve_scanned_ocr_language() == "gujarati"
            else _display_language(validation_report.get("languages", ["unknown"])[0])
        ),
        tables=table_models,
        pages=page_models,
        metadata=metadata,
    )

    logger.info(
        "[%s] Pipeline complete | pages=%s | tables=%s | confidence=%.3f | time=%ss",
        job_id,
        metadata.pages,
        len(table_models),
        metadata.confidence_score,
        processing_time,
    )

    return result


def save_result_to_disk(result: ExtractionResult, output_path: Path) -> None:
    """Persist extraction result as JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump()
    payload["expires_at"] = (
        datetime.now(timezone.utc) + timedelta(seconds=settings.RESULT_EXPIRES_SECONDS)
    ).isoformat()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    logger.info("Result saved to %s", output_path)
