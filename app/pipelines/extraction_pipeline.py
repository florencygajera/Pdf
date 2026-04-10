"""
Extraction Pipeline — Master Orchestrator
Coordinates all services in the correct order:

  1. PDF Type Detection
  2. Digital OR OCR Extraction (per page)
  3. Table Extraction (per page)
  4. Layout Reconstruction
  5. Noise Cleaning
  6. Validation & Scoring
  7. Structured Output Assembly

Handles mixed PDFs (some pages digital, some scanned) transparently.
Large file support via chunked page processing.
"""

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.config.constants import (
    CHUNK_SIZE_PAGES,
    PDF_TYPE_DIGITAL,
    PDF_TYPE_SCANNED,
    STATE_DONE,
    STATE_FAILED,
    STATE_PROCESSING,
)
from app.models.response_model import (
    ExtractionMetadata,
    ExtractionResult,
    PageResult,
    TableData,
)
from app.services.digital_extractor import extract_digital_pdf
from app.services.layout_engine import reconstruct_reading_order
from app.services.noise_cleaner import clean_pages, clean_text_block
from app.services.ocr_extractor import extract_ocr_pdf
from app.services.pdf_detector import DocumentClassification, detect_pdf_type
from app.services.table_extractor import extract_tables_digital, extract_tables_scanned
from app.services.validator import validate_extraction_result
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _chunk_list(lst: List, size: int) -> List[List]:
    """Split a list into chunks of given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


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


def _process_digital_pages(
    pdf_path: Path,
    doc_classification: DocumentClassification,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Process all digital pages — extraction + table detection."""
    digital_page_nums = [
        p.page_number
        for p in doc_classification.pages
        if p.pdf_type == PDF_TYPE_DIGITAL
    ]

    if not digital_page_nums:
        return []

    logger.info(f"Processing {len(digital_page_nums)} digital pages...")
    chunks = _chunk_list(digital_page_nums, CHUNK_SIZE_PAGES)
    all_results = []

    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"Digital chunk {chunk_idx + 1}/{len(chunks)} | pages {chunk}")
        page_data = extract_digital_pdf(pdf_path, page_numbers=chunk)

        for page_info in page_data:
            page_num = page_info["page_number"]
            # Reconstruct layout (handles multi-column)
            if page_info.get("blocks"):
                ordered_blocks = reconstruct_reading_order(
                    page_info["blocks"],
                    page_width=612.0,
                )
                # Rebuild text from re-ordered blocks
                page_info["text"] = "\n\n".join(
                    b["text"] for b in ordered_blocks if b.get("text")
                )

            # Extract tables for this page
            try:
                tables = extract_tables_digital(pdf_path, page_num)
            except Exception as exc:
                logger.warning(f"Table extraction failed for page {page_num}: {exc}")
                tables = []

            page_info["tables"] = tables
            page_info["confidence"] = 0.95  # Digital = high confidence
            all_results.append(page_info)

        if progress_callback:
            progress_callback(chunk_idx + 1, len(chunks), "digital")

    return all_results


def _process_scanned_pages(
    pdf_path: Path,
    doc_classification: DocumentClassification,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Process all scanned pages — OCR + table grid detection."""
    scanned_page_nums = [
        p.page_number
        for p in doc_classification.pages
        if p.pdf_type == PDF_TYPE_SCANNED
    ]

    if not scanned_page_nums:
        return []

    logger.info(f"Processing {len(scanned_page_nums)} scanned pages via OCR...")
    chunks = _chunk_list(scanned_page_nums, CHUNK_SIZE_PAGES)
    all_results = []

    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"OCR chunk {chunk_idx + 1}/{len(chunks)} | pages {chunk}")
        ocr_data = extract_ocr_pdf(pdf_path, page_numbers=chunk)

        for page_info in ocr_data:
            page_num = page_info["page_number"]

            # Table detection from OCR results
            raw_ocr = page_info.get("raw_results", [])
            tables = []
            if raw_ocr:
                import numpy as np

                # Use a blank array — actual image used internally in scanned extractor
                # would need to pass preprocessed image; use raw_results for cell mapping
                dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                try:
                    tables = extract_tables_scanned(dummy_img, raw_ocr, page_num)
                except Exception as exc:
                    logger.warning(
                        f"Scanned table extraction failed page {page_num}: {exc}"
                    )

            page_info["tables"] = tables
            all_results.append(page_info)

        if progress_callback:
            progress_callback(chunk_idx + 1, len(chunks), "ocr")

    return all_results


def run_extraction_pipeline(
    pdf_path: Path,
    job_id: str,
    progress_callback: Optional[Callable] = None,
) -> ExtractionResult:
    """
    Master pipeline: detect → extract → clean → validate → output.

    Args:
        pdf_path: Path to uploaded PDF.
        job_id: Unique job identifier.
        progress_callback: Optional fn(step, total, stage_name) for progress tracking.

    Returns:
        ExtractionResult Pydantic model.

    Raises:
        ValueError, RuntimeError on unrecoverable errors.
    """
    start_time = time.time()
    warnings: List[str] = []

    # ── Step 1: PDF Type Detection ────────────────────────────────────────────
    logger.info(f"[{job_id}] Step 1: PDF type detection")
    try:
        doc_classification = detect_pdf_type(pdf_path)
    except ValueError as exc:
        raise ValueError(f"PDF validation failed: {exc}") from exc

    # ── Step 2: Extraction (digital + scanned) ────────────────────────────────
    logger.info(f"[{job_id}] Step 2: Extracting content")
    digital_results = _process_digital_pages(
        pdf_path, doc_classification, progress_callback
    )
    scanned_results = _process_scanned_pages(
        pdf_path, doc_classification, progress_callback
    )

    # Merge and sort all page results by page_number
    all_page_results: List[Dict] = sorted(
        digital_results + scanned_results,
        key=lambda p: p.get("page_number", 0),
    )

    # ── Step 3: Noise Cleaning ────────────────────────────────────────────────
    logger.info(f"[{job_id}] Step 3: Noise removal")
    page_texts = [p.get("text", "") for p in all_page_results]
    cleaned_texts = clean_pages(page_texts)
    for i, page in enumerate(all_page_results):
        page["text"] = (
            cleaned_texts[i] if i < len(cleaned_texts) else page.get("text", "")
        )

    # ── Step 4: Aggregate all tables ──────────────────────────────────────────
    all_tables: List[Dict] = []
    for page in all_page_results:
        all_tables.extend(page.get("tables", []))

    # ── Step 5: Validation ────────────────────────────────────────────────────
    logger.info(f"[{job_id}] Step 4: Validation pass")
    validation_report = validate_extraction_result(all_page_results, all_tables)
    warnings.extend(
        f"Page {pn}: {w}"
        for pn, ws in validation_report.get("page_warnings", {}).items()
        for w in ws
    )

    # ── Step 6: Assemble Output ───────────────────────────────────────────────
    logger.info(f"[{job_id}] Step 5: Assembling output")

    # Full merged text
    full_text = "\n\n---\n\n".join(
        f"[Page {p.get('page_number', i + 1)}]\n{p.get('text', '').strip()}"
        for i, p in enumerate(all_page_results)
        if p.get("text", "").strip()
    )
    full_text = clean_text_block(full_text)

    # Per-page models
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

    # Table models
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

    metadata = ExtractionMetadata(
        pages=doc_classification.total_pages,
        pdf_type=doc_classification.overall_type,
        confidence_score=validation_report["overall_confidence"],
        processing_time_seconds=processing_time,
        languages_detected=validation_report.get("languages", []),
        ocr_engine="PaddleOCR" if doc_classification.scanned_page_count > 0 else None,
        warnings=warnings[:50],  # Cap warning list
    )

    result = ExtractionResult(
        job_id=job_id,
        status=STATE_DONE,
        text=full_text,
        tables=table_models,
        pages=page_models,
        metadata=metadata,
    )

    logger.info(
        f"[{job_id}] Pipeline complete | "
        f"pages={metadata.pages} | tables={len(table_models)} | "
        f"confidence={metadata.confidence_score:.3f} | "
        f"time={processing_time}s"
    )

    return result


def save_result_to_disk(result: ExtractionResult, output_path: Path) -> None:
    """Persist extraction result as JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info(f"Result saved to {output_path}")
