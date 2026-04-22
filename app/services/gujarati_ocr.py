"""
Gujarati OCR engine built on Tesseract.

This module is optimized for scanned Gujarati PDFs:
- parallel per-page processing
- grayscale-only rendering path
- grayscale + denoise + threshold preprocessing
- two-pass OCR only: (guj+eng, PSM 6) then retry empty pages with (guj+eng, PSM 11)
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
from PIL import Image

from app.config.settings import settings
from app.utils.image_preprocessing import adaptive_threshold, pil_to_cv2, remove_noise
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TESSERACT_AVAILABLE: Optional[bool] = None
_TESSERACT_CHECK_LOCK = Lock()

_PRIMARY_PSM = 6
_RETRY_PSM = 11
_TARGET_CONFIDENCE = 0.6
_LOW_CONFIDENCE_WARNING = 0.35
_EMPTY_CONFIDENCE_CUTOFF = 0.2

_OCR_LANG = "guj+eng"


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

    bgr = pil_to_cv2(image)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _preprocess_for_gujarati(image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
    """
    Gujarati-safe preprocessing:
    - grayscale only
    - denoise
    - adaptive threshold tuned for Gujarati scans
    """
    gray = _as_grayscale_array(image)
    gray = remove_noise(gray)
    binary = adaptive_threshold(gray)
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
    best_psm = _PRIMARY_PSM
    best_conf = 0.0

    try:
        primary_text, primary_conf = _run_tesseract(
            preprocessed, lang=_OCR_LANG, psm=_PRIMARY_PSM
        )
    except Exception as exc:
        logger.warning(
            "Tesseract failed | page=%s | lang=%s | psm=%s | error=%s",
            page_number,
            _OCR_LANG,
            _PRIMARY_PSM,
            exc,
        )
        warnings.append(f"Tesseract error (lang={_OCR_LANG} psm={_PRIMARY_PSM}): {exc}")
        primary_text, primary_conf = "", 0.0

    primary_score = _score_result(primary_text, primary_conf)
    logger.info(
        "[OCR] Page %s -> Engine: Tesseract | lang=%s | psm=%s | confidence=%.3f | score=%.3f | chars=%s",
        page_number,
        _OCR_LANG,
        _PRIMARY_PSM,
        primary_conf,
        primary_score,
        len(primary_text),
    )

    best_text = primary_text
    best_score = primary_score
    best_conf = primary_conf

    if not primary_text.strip():
        try:
            retry_text, retry_conf = _run_tesseract(
                preprocessed, lang=_OCR_LANG, psm=_RETRY_PSM
            )
        except Exception as exc:
            logger.warning(
                "Tesseract retry failed | page=%s | lang=%s | psm=%s | error=%s",
                page_number,
                _OCR_LANG,
                _RETRY_PSM,
                exc,
            )
            warnings.append(
                f"Tesseract error (lang={_OCR_LANG} psm={_RETRY_PSM}): {exc}"
            )
            retry_text, retry_conf = "", 0.0

        retry_score = _score_result(retry_text, retry_conf)
        logger.info(
            "[OCR] Page %s -> Engine: Tesseract | lang=%s | psm=%s | confidence=%.3f | score=%.3f | chars=%s",
            page_number,
            _OCR_LANG,
            _RETRY_PSM,
            retry_conf,
            retry_score,
            len(retry_text),
        )
        if retry_score > best_score:
            best_text = retry_text
            best_score = retry_score
            best_conf = retry_conf
            best_psm = _RETRY_PSM

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
        _OCR_LANG,
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


def _render_page_to_pil(
    page: "fitz.Page",
    dpi: int,
) -> Image.Image:
    scale = max(1.0, dpi / 72.0)
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if mode == "RGBA":
        image = image.convert("RGB")
    return image


def _process_rendered_page(
    image: Image.Image,
    page_number: int,
    dpi: int,
) -> Dict[str, Any]:
    return ocr_gujarati_page(image, page_number=page_number, dpi=dpi)


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
    dpi = max(300, dpi, settings.OCR_DPI)

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
            (page_num, _render_page_to_pil(doc[page_num - 1], dpi=dpi))
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
            executor.submit(_process_rendered_page, image, page_num, dpi): page_num
            for page_num, image in rendered_pages
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
