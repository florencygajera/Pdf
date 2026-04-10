"""
Extraction Pipeline — Master Orchestrator

Performance-focused version:
  - reads the PDF once into memory
  - classifies from in-memory bytes
  - runs digital and OCR branches concurrently
  - fans out chunk processing with ThreadPoolExecutor
  - parallelizes table extraction per page
  - preserves stable output ordering and existing API contracts

Designed to stay safe under Windows Celery thread pools.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from app.services.layout_engine import reconstruct_reading_order
from app.services.noise_cleaner import clean_pages, clean_text_block
from app.services.ocr_extractor import extract_ocr_pdf_from_bytes
from app.services.pdf_detector import DocumentClassification, detect_pdf_type_from_bytes
from app.services.table_extractor import extract_tables_digital, extract_tables_scanned
from app.services.validator import validate_extraction_result
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _chunk_list(lst: List, size: int) -> List[List]:
    """Split a list into chunks of given size."""
    if size <= 0:
        return [lst]
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _choose_chunk_size(total_pages: int, base: int = 10) -> int:
    """
    Pick a larger chunk size for long documents to reduce executor overhead.
    More pages -> fewer, larger chunks.
    """
    if total_pages <= 12:
        return max(4, base)
    if total_pages <= 40:
        return max(8, base)
    return min(25, max(base, total_pages // 4 or base))


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


def _process_digital_chunk(
    pdf_path: Path,
    pdf_bytes: bytes,
    chunk: List[int],
) -> List[Dict[str, Any]]:
    """
    Process a chunk of digital pages.

    The chunk worker stays self-contained so it can safely run on a Windows
    thread pool. Using in-memory bytes avoids re-reading the PDF from disk.
    """
    page_data = extract_digital_pdf(pdf_path, page_numbers=chunk, pdf_bytes=pdf_bytes)
    for page_info in page_data:
        if page_info.get("blocks"):
            ordered_blocks = reconstruct_reading_order(
                page_info["blocks"],
                page_width=612.0,
            )
            page_info["text"] = "\n\n".join(
                b["text"] for b in ordered_blocks if b.get("text")
            )
    return page_data


def _extract_digital_tables_for_page(
    pdf_path: Path,
    pdf_bytes: bytes,
    page_num: int,
) -> List[Dict[str, Any]]:
    try:
        return extract_tables_digital(pdf_path, page_num, pdf_bytes=pdf_bytes)
    except Exception as exc:
        logger.warning("Table extraction failed for page %s: %s", page_num, exc)
        return []


def _process_digital_pages(
    pdf_path: Path,
    doc_classification: DocumentClassification,
    progress_callback: Optional[Callable] = None,
    pdf_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    """Process all digital pages using concurrent chunks and page-level table extraction."""
    digital_page_nums = [
        p.page_number
        for p in doc_classification.pages
        if p.pdf_type == PDF_TYPE_DIGITAL
    ]

    if not digital_page_nums:
        return []

    pdf_bytes = pdf_bytes or pdf_path.read_bytes()
    chunk_size = _choose_chunk_size(len(digital_page_nums))
    chunks = _chunk_list(digital_page_nums, chunk_size)
    all_results: List[Dict[str, Any]] = []
    progress_lock = Lock()
    chunk_workers = min(4, len(chunks)) if len(chunks) > 1 else 1

    logger.info(
        "Processing %s digital pages in %s chunks | chunk_size=%s | workers=%s",
        len(digital_page_nums),
        len(chunks),
        chunk_size,
        chunk_workers,
    )

    with ThreadPoolExecutor(max_workers=chunk_workers) as executor:
        futures = {
            executor.submit(_process_digital_chunk, pdf_path, pdf_bytes, chunk): idx
            for idx, chunk in enumerate(chunks)
        }

        for completed_idx, future in enumerate(as_completed(futures), start=1):
            chunk_results = future.result()

            # Parallelize table extraction per page. This is usually cheaper than OCR
            # and benefits nicely from thread pooling.
            table_workers = min(4, len(chunk_results)) if len(chunk_results) > 1 else 1
            if table_workers > 1:
                with ThreadPoolExecutor(max_workers=table_workers) as table_pool:
                    table_futures = {
                        table_pool.submit(
                            _extract_digital_tables_for_page,
                            pdf_path,
                            pdf_bytes,
                            page_info["page_number"],
                        ): page_info
                        for page_info in chunk_results
                    }
                    for table_future in as_completed(table_futures):
                        page_info = table_futures[table_future]
                        page_info["tables"] = table_future.result()
                        page_info["confidence"] = 0.95
                        all_results.append(page_info)
            else:
                for page_info in chunk_results:
                    page_info["tables"] = _extract_digital_tables_for_page(
                        pdf_path, pdf_bytes, page_info["page_number"]
                    )
                    page_info["confidence"] = 0.95
                    all_results.append(page_info)

            _run_with_progress_lock(
                progress_callback,
                progress_lock,
                completed_idx,
                len(chunks),
                "digital",
            )

    all_results.sort(key=lambda p: p.get("page_number", 0))
    return all_results


def _process_scanned_chunk(
    pdf_bytes: bytes,
    chunk: List[int],
    dpi: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return extract_ocr_pdf_from_bytes(
        pdf_bytes,
        page_numbers=chunk,
        dpi=dpi or settings.effective_ocr_dpi(len(chunk)),
    )


def _process_scanned_pages(
    pdf_path: Path,
    doc_classification: DocumentClassification,
    progress_callback: Optional[Callable] = None,
    pdf_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    """Process scanned pages using concurrent OCR chunks."""
    scanned_page_nums = [
        p.page_number
        for p in doc_classification.pages
        if p.pdf_type == PDF_TYPE_SCANNED
    ]

    if not scanned_page_nums:
        return []

    pdf_bytes = pdf_bytes or pdf_path.read_bytes()
    chunk_size = settings.effective_ocr_chunk_size(len(scanned_page_nums))
    chunks = _chunk_list(scanned_page_nums, chunk_size)
    all_results: List[Dict[str, Any]] = []
    progress_lock = Lock()
    chunk_workers = min(settings.effective_ocr_chunk_workers, len(chunks)) if len(chunks) > 1 else 1

    logger.info(
        "Processing %s scanned pages via OCR in %s chunks | chunk_size=%s | workers=%s | page_workers=%s",
        len(scanned_page_nums),
        len(chunks),
        chunk_size,
        chunk_workers,
        settings.effective_ocr_page_workers,
    )

    with ThreadPoolExecutor(max_workers=chunk_workers) as executor:
        futures = {
            executor.submit(
                _process_scanned_chunk,
                pdf_bytes,
                chunk,
                settings.effective_ocr_dpi(len(chunk)),
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        for completed_idx, future in enumerate(as_completed(futures), start=1):
            ocr_data = future.result()

            # Scanned table detection is lightweight, but still parallelizable by page.
            table_workers = min(4, len(ocr_data)) if len(ocr_data) > 1 else 1
            if table_workers > 1:
                with ThreadPoolExecutor(max_workers=table_workers) as table_pool:
                    table_futures = {}
                    for page_info in ocr_data:
                        raw_ocr = page_info.get("raw_results", [])
                        if not raw_ocr:
                            page_info["tables"] = []
                            all_results.append(page_info)
                            continue
                        import numpy as np

                        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                        table_futures[
                            table_pool.submit(
                                extract_tables_scanned,
                                dummy_img,
                                raw_ocr,
                                page_info["page_number"],
                            )
                        ] = page_info

                    for table_future in as_completed(table_futures):
                        page_info = table_futures[table_future]
                        page_info["tables"] = table_future.result()
                        all_results.append(page_info)
            else:
                for page_info in ocr_data:
                    raw_ocr = page_info.get("raw_results", [])
                    if raw_ocr:
                        import numpy as np

                        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                        try:
                            page_info["tables"] = extract_tables_scanned(
                                dummy_img, raw_ocr, page_info["page_number"]
                            )
                        except Exception as exc:
                            logger.warning(
                                "Scanned table extraction failed page %s: %s",
                                page_info["page_number"],
                                exc,
                            )
                            page_info["tables"] = []
                    else:
                        page_info["tables"] = []
                    all_results.append(page_info)

            _run_with_progress_lock(
                progress_callback,
                progress_lock,
                completed_idx,
                len(chunks),
                "ocr",
            )

    all_results.sort(key=lambda p: p.get("page_number", 0))
    return all_results


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
        doc_classification = detect_pdf_type_from_bytes(pdf_bytes, file_name=pdf_path.name)
    except ValueError as exc:
        raise ValueError(f"PDF validation failed: {exc}") from exc

    logger.info("[%s] Step 2: Extracting content", job_id)
    digital_results: List[Dict[str, Any]] = []
    scanned_results: List[Dict[str, Any]] = []

    # Digital and OCR branches are independent, so we run them concurrently.
    branch_futures = {}
    max_branch_workers = 2
    with ThreadPoolExecutor(max_workers=max_branch_workers) as executor:
        branch_futures[executor.submit(
            _process_digital_pages,
            pdf_path,
            doc_classification,
            progress_callback,
            pdf_bytes,
        )] = "digital"
        branch_futures[executor.submit(
            _process_scanned_pages,
            pdf_path,
            doc_classification,
            progress_callback,
            pdf_bytes,
        )] = "scanned"

        for future in as_completed(branch_futures):
            kind = branch_futures[future]
            if kind == "digital":
                digital_results = future.result()
            else:
                scanned_results = future.result()

    all_page_results: List[Dict] = sorted(
        digital_results + scanned_results,
        key=lambda p: p.get("page_number", 0),
    )

    logger.info("[%s] Step 3: Noise removal", job_id)
    clean_indices: List[int] = []
    clean_texts: List[str] = []
    for idx, page in enumerate(all_page_results):
        text = page.get("text", "")
        if page.get("confidence", 0.0) >= 0.9 and text.strip():
            continue
        clean_indices.append(idx)
        clean_texts.append(text)

    if clean_texts:
        cleaned_texts = clean_pages(clean_texts)
        for idx, cleaned in zip(clean_indices, cleaned_texts):
            all_page_results[idx]["text"] = cleaned

    all_tables: List[Dict] = []
    for page in all_page_results:
        all_tables.extend(page.get("tables", []))

    logger.info("[%s] Step 4: Validation pass", job_id)
    validation_report = validate_extraction_result(all_page_results, all_tables)
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

    metadata = ExtractionMetadata(
        pages=doc_classification.total_pages,
        pdf_type=doc_classification.overall_type,
        confidence_score=validation_report["overall_confidence"],
        processing_time_seconds=processing_time,
        languages_detected=validation_report.get("languages", []),
        ocr_engine="PaddleOCR" if doc_classification.scanned_page_count > 0 else None,
        warnings=warnings[:50],
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info("Result saved to %s", output_path)
