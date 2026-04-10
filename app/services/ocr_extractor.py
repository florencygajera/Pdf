"""
OCR Extraction Engine
Converts scanned PDF pages to images, preprocesses them, runs PaddleOCR,
and returns sorted, confidence-filtered text.

PaddleOCR is preferred for its superior accuracy on printed documents.
Falls back gracefully if PaddleOCR is unavailable.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError

from app.config.constants import LINE_Y_TOLERANCE_OCR
from app.config.settings import settings
from app.utils.image_preprocessing import preprocess_page_image
from app.utils.logger import get_logger
from app.utils.sorting import (
    merge_hyphenated_lines,
    sort_ocr_results,
)

logger = get_logger(__name__)

# ── PaddleOCR singleton (loaded once, reused across calls) ────────────────────
_paddle_ocr_instance = None


def _get_paddle_ocr():
    """
    Lazy-load PaddleOCR. GPU is controlled by settings.OCR_USE_GPU.
    Returns None if PaddleOCR is not installed.
    """
    global _paddle_ocr_instance
    if _paddle_ocr_instance is not None:
        return _paddle_ocr_instance

    try:
        from paddleocr import PaddleOCR

        lang = settings.OCR_LANGUAGES[0] if settings.OCR_LANGUAGES else "en"
        _paddle_ocr_instance = PaddleOCR(
            use_angle_cls=True,  # detect rotated text
            lang=lang,
            use_gpu=settings.OCR_USE_GPU,
            show_log=False,
        )
        logger.info(f"PaddleOCR initialized | lang={lang} | gpu={settings.OCR_USE_GPU}")
        return _paddle_ocr_instance

    except ImportError:
        logger.warning(
            "PaddleOCR not installed. Install via: pip install paddlepaddle paddleocr"
        )
        return None


def _run_paddle_ocr(image_array: np.ndarray) -> List:
    """
    Run PaddleOCR on a preprocessed numpy image.
    Returns raw PaddleOCR result list or empty list on failure.
    """
    ocr = _get_paddle_ocr()
    if ocr is None:
        raise RuntimeError(
            "PaddleOCR is not available. Install: pip install paddlepaddle paddleocr"
        )

    try:
        result = ocr.ocr(image_array, cls=True)
        # PaddleOCR returns [[lines...]] — flatten one level
        if result and isinstance(result[0], list):
            return result[0]
        return result or []
    except Exception as exc:
        logger.error(f"PaddleOCR inference failed: {exc}", exc_info=True)
        return []


def _filter_by_confidence(
    ocr_results: List,
    threshold: float,
) -> List:
    """Remove OCR results below the confidence threshold."""
    filtered = []
    for item in ocr_results:
        if not item or len(item) < 2:
            continue
        _, (text, conf) = item
        if conf >= threshold:
            filtered.append(item)
        else:
            logger.debug(f"Filtered low-confidence token '{text}' ({conf:.2f})")
    return filtered


def _ocr_results_to_text(ocr_results: List) -> str:
    """
    Convert sorted+filtered PaddleOCR results to a clean text string.
    Merges tokens on the same line, separates lines with newlines.
    """
    if not ocr_results:
        return ""

    sorted_results = sort_ocr_results(ocr_results)

    def top_y(r):
        return min(pt[1] for pt in r[0])

    # Group tokens into lines using y-tolerance
    lines: List[List[str]] = []
    current_line: List[str] = []
    current_y: Optional[float] = None

    for item in sorted_results:
        box, (text, _) = item
        y = top_y(item)
        if current_y is None or abs(y - current_y) > LINE_Y_TOLERANCE_OCR:
            if current_line:
                lines.append(current_line)
            current_line = [text]
            current_y = y
        else:
            current_line.append(text)

    if current_line:
        lines.append(current_line)

    joined_lines = [" ".join(ln) for ln in lines]
    joined_lines = merge_hyphenated_lines(joined_lines)
    return "\n".join(joined_lines)


def _compute_page_confidence(ocr_results: List) -> float:
    """Average confidence of all accepted OCR tokens."""
    if not ocr_results:
        return 0.0
    confs = []
    for item in ocr_results:
        if item and len(item) >= 2:
            _, (_, conf) = item
            confs.append(conf)
    return sum(confs) / len(confs) if confs else 0.0


def ocr_single_page_image(
    image,  # PIL Image
    page_number: int,
) -> Dict[str, Any]:
    """
    Run the full OCR pipeline on a single page PIL image.
    Returns dict: {page_number, text, confidence, warnings, raw_results}
    """
    warnings: List[str] = []

    # 1. Preprocess
    try:
        processed_arr, preprocess_meta = preprocess_page_image(image)
    except Exception as exc:
        logger.error(
            f"Preprocessing failed on page {page_number}: {exc}", exc_info=True
        )
        warnings.append(f"Preprocessing failed: {exc}")
        # Fall back to raw image
        import cv2

        processed_arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        preprocess_meta = {}

    if preprocess_meta.get("deskew_angle", 0) != 0:
        warnings.append(f"Page was deskewed by {preprocess_meta['deskew_angle']:.1f}°")

    # 2. OCR
    try:
        raw_results = _run_paddle_ocr(processed_arr)
    except Exception as exc:
        logger.error(f"OCR failed on page {page_number}: {exc}", exc_info=True)
        return {
            "page_number": page_number,
            "text": "",
            "confidence": 0.0,
            "warnings": [f"OCR engine error: {exc}"],
            "raw_results": [],
        }

    if not raw_results:
        warnings.append(
            f"Page {page_number}: OCR returned no results (blank or unreadable)."
        )
        return {
            "page_number": page_number,
            "text": "",
            "confidence": 0.0,
            "warnings": warnings,
            "raw_results": [],
        }

    # 3. Confidence filter
    filtered = _filter_by_confidence(raw_results, settings.OCR_CONFIDENCE_THRESHOLD)
    confidence = _compute_page_confidence(filtered)

    # 4. Text reconstruction
    text = _ocr_results_to_text(filtered)

    return {
        "page_number": page_number,
        "text": text,
        "confidence": confidence,
        "warnings": warnings,
        "raw_results": filtered,
    }


def extract_ocr_pdf(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    dpi: int = None,
) -> List[Dict[str, Any]]:
    """
    Full OCR extraction pipeline for scanned PDF pages.

    Args:
        pdf_path: Path to PDF.
        page_numbers: 1-indexed pages to process. None = all.
        dpi: Resolution for PDF→image conversion (default: settings.OCR_DPI).

    Returns:
        List of per-page result dicts.
    """
    dpi = dpi or settings.OCR_DPI
    results: List[Dict[str, Any]] = []

    # ── Convert PDF → images ───────────────────────────────────────────────
    try:
        all_images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt="PNG",
            thread_count=2,
            first_page=min(page_numbers) if page_numbers else None,
            last_page=max(page_numbers) if page_numbers else None,
        )
    except PDFSyntaxError as exc:
        raise ValueError(f"PDF syntax error during image conversion: {exc}") from exc
    except PDFPageCountError as exc:
        raise ValueError(f"Cannot count PDF pages: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"pdf2image conversion failed: {exc}") from exc

    target_set = set(page_numbers) if page_numbers else None

    for idx, pil_image in enumerate(all_images):
        page_num = (min(page_numbers) if page_numbers else 1) + idx

        if target_set and page_num not in target_set:
            continue

        logger.info(f"OCR processing page {page_num} ({dpi} DPI)...")
        result = ocr_single_page_image(pil_image, page_num)
        results.append(result)

        # Free memory immediately — large images are expensive
        del pil_image
        gc.collect()

    logger.info(
        f"OCR extraction complete | pages={len(results)} | file={pdf_path.name}"
    )
    return results
