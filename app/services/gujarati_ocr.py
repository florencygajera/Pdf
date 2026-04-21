"""
Gujarati OCR engine built on Tesseract.

This module is optimized for scanned Gujarati PDFs:
- parallel per-page processing
- grayscale-only rendering path
- CLAHE + median blur + Gujarati-friendly thresholding
- two-pass OCR only: (guj+eng, PSM 6) then (guj, PSM 6)
- early stop when confidence is high enough

Each page returns the existing pipeline schema:
{
    "page_number": int,
    "text": str,
    "confidence": float,
    "warnings": list,
    "tables": list
}
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import fitz
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

_TESSERACT_AVAILABLE: Optional[bool] = None
_TESSERACT_CHECK_LOCK = Lock()

_PSM_BLOCK = 6
_TARGET_CONFIDENCE = 0.6
_LOW_CONFIDENCE_WARNING = 0.35
_EMPTY_CONFIDENCE_CUTOFF = 0.2

_OCR_VARIANTS: Tuple[Tuple[str, int], ...] = (
    ("guj+eng", _PSM_BLOCK),
    ("guj", _PSM_BLOCK),
)


def _check_tesseract_available() -> bool:
    """Check whether pytesseract and the Gujarati language pack are available."""
    try:
        import pytesseract

        langs = pytesseract.get_languages(config="")
        return "guj" in langs
    except Exception as exc:
        logger.warning("Tesseract unavailable: %s", exc)
        return False


def tesseract_available() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE

    with _TESSERACT_CHECK_LOCK:
        if _TESSERACT_AVAILABLE is None:
            _TESSERACT_AVAILABLE = _check_tesseract_available()
    return _TESSERACT_AVAILABLE


def _as_grayscale_array(image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
    """Convert input to a single grayscale numpy array exactly once."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        raise ValueError("Unsupported ndarray shape for Gujarati OCR preprocessing.")

    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    raise ValueError("Unsupported image shape for Gujarati OCR preprocessing.")


def _preprocess_for_gujarati(image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
    """
    Gujarati-safe preprocessing:
    - grayscale only
    - CLAHE for low contrast
    - median blur, not Gaussian
    - adaptive threshold tuned for shirorekha
    - very light dilation only
    """
    gray = _as_grayscale_array(image)

    h, w = gray.shape[:2]
    if h > 0 and h < 1400:
        scale = 1400 / h
        gray = cv2.resize(
            gray,
            (max(1, int(w * scale)), 1400),
            interpolation=cv2.INTER_CUBIC,
        )

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        17,
        5,
    )

    # Keep matras/shirorekha intact: avoid erosion, use only a tiny dilation.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def _run_tesseract(
    image: np.ndarray,
    lang: str,
    psm: int,
) -> Tuple[str, float]:
    """
    Run Tesseract on a preprocessed image and return:
    - extracted text
    - mean confidence from OCR data
    """
    import pytesseract

    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(image, lang=lang, config=config).strip()

    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        confs = []
        for raw in data.get("conf", []):
            try:
                conf = float(raw)
            except Exception:
                continue
            if conf >= 0:
                confs.append(conf / 100.0 if conf > 1 else conf)
        confidence = float(np.mean(confs)) if confs else 0.0
    except Exception:
        confidence = 0.0

    return text, confidence


def _score_result(text: str, confidence: float) -> float:
    """
    Blend OCR confidence with Gujarati-script coverage so we keep useful
    results even when Tesseract confidence is imperfect on ligatures.
    """
    if not text.strip():
        return 0.0

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0

    guj_chars = sum(1 for c in chars if "\u0a80" <= c <= "\u0aff")
    latin_chars = sum(1 for c in chars if c.isascii() and c.isalpha())
    digit_chars = sum(1 for c in chars if c.isdigit())

    readable_ratio = (guj_chars + latin_chars + digit_chars) / len(chars)
    script_ratio = guj_chars / len(chars)

    score = (confidence * 0.7) + (readable_ratio * 0.2) + (script_ratio * 0.1)
    return round(max(0.0, min(score, 1.0)), 4)


def _ocr_page_with_fallbacks(
    image: Union[np.ndarray, "Image.Image"],
    page_number: int,
    dpi: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    warnings: List[str] = []

    if not tesseract_available():
        warnings.append(
            "Tesseract/guj not installed. Install tesseract-ocr tesseract-ocr-guj."
        )
        return {
            "page_number": page_number,
            "text": "",
            "confidence": 0.0,
            "warnings": warnings,
            "tables": [],
        }

    try:
        preprocessed = _preprocess_for_gujarati(image)
    except Exception as exc:
        logger.warning(
            "Gujarati preprocessing fallback | page=%s | error=%s",
            page_number,
            exc,
        )
        warnings.append(f"Preprocessing fallback used: {exc}")
        preprocessed = _as_grayscale_array(image)

    best_text = ""
    best_score = 0.0
    best_lang = ""
    best_psm = _PSM_BLOCK
    best_conf = 0.0

    for lang, psm in _OCR_VARIANTS:
        try:
            text, confidence = _run_tesseract(preprocessed, lang=lang, psm=psm)
        except Exception as exc:
            logger.warning(
                "Tesseract failed | page=%s | lang=%s | psm=%s | error=%s",
                page_number,
                lang,
                psm,
                exc,
            )
            warnings.append(f"Tesseract error (lang={lang} psm={psm}): {exc}")
            continue

        score = _score_result(text, confidence)
        logger.info(
            "Gujarati OCR pass | page=%s | lang=%s | psm=%s | confidence=%.3f | score=%.3f | chars=%s",
            page_number,
            lang,
            psm,
            confidence,
            score,
            len(text),
        )

        if score > best_score:
            best_text = text
            best_score = score
            best_lang = lang
            best_psm = psm
            best_conf = confidence

        if score >= _TARGET_CONFIDENCE:
            break

    if best_score < _EMPTY_CONFIDENCE_CUTOFF:
        best_text = ""

    if best_score < _LOW_CONFIDENCE_WARNING:
        warnings.append("Low OCR confidence")

    if not best_text.strip():
        warnings.append(f"Page {page_number}: Tesseract returned no text.")

    elapsed = time.perf_counter() - started
    logger.info(
        "Gujarati OCR page done | page=%s | lang=%s | psm=%s | confidence=%.3f | score=%.3f | duration_seconds=%.2f",
        page_number,
        best_lang or "n/a",
        best_psm,
        best_conf,
        best_score,
        elapsed,
    )

    return {
        "page_number": page_number,
        "text": best_text,
        "confidence": float(best_score),
        "warnings": warnings,
        "tables": [],
    }


def ocr_gujarati_page(
    image: Union[np.ndarray, "Image.Image"],
    page_number: int,
    dpi: int = 250,
) -> Dict[str, Any]:
    """OCR a single page image using the Gujarati Tesseract path."""
    return _ocr_page_with_fallbacks(image=image, page_number=page_number, dpi=dpi)


def _render_page_to_gray(
    page: "fitz.Page",
    dpi: int,
) -> np.ndarray:
    scale = max(1.5, dpi / 72.0)
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return gray.copy()


def _process_rendered_page(
    gray: np.ndarray,
    page_number: int,
    dpi: int,
) -> Dict[str, Any]:
    return ocr_gujarati_page(gray, page_number=page_number, dpi=dpi)


def _parallel_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu))


def extract_gujarati_pdf(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    dpi: int = 250,
    pdf_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    """
    Extract Gujarati OCR page results in parallel.

    Page rendering is done at a minimum of 250 DPI for Gujarati accuracy.
    """
    import fitz

    started = time.perf_counter()
    dpi = max(250, dpi)

    try:
        doc = (
            fitz.open(stream=pdf_bytes, filetype="pdf")
            if pdf_bytes is not None
            else fitz.open(str(pdf_path))
        )
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {exc}") from exc

    try:
        total = len(doc)
        targets = page_numbers or list(range(1, total + 1))
        targets = [p for p in targets if 1 <= p <= total]
        rendered_pages = [
            (page_num, _render_page_to_gray(doc[page_num - 1], dpi=dpi))
            for page_num in targets
        ]
    finally:
        doc.close()

    if not targets:
        return []

    workers = min(_parallel_workers(), len(targets))
    logger.info(
        "Gujarati OCR batch start | file=%s | pages=%s | workers=%s | dpi=%s",
        pdf_path.name,
        len(targets),
        workers,
        dpi,
    )

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_rendered_page, gray, page_num, dpi): page_num
            for page_num, gray in rendered_pages
        }
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.warning(
                    "Gujarati OCR page failed | page=%s | error=%s",
                    page_num,
                    exc,
                )
                results.append(
                    {
                        "page_number": page_num,
                        "text": "",
                        "confidence": 0.0,
                        "warnings": [f"OCR failed: {exc}"],
                        "tables": [],
                    }
                )

    results.sort(key=lambda r: r["page_number"])
    logger.info(
        "Gujarati OCR batch end | file=%s | pages=%s | duration_seconds=%.2f",
        pdf_path.name,
        len(results),
        time.perf_counter() - started,
    )
    return results
