"""
Gujarati OCR Engine using Tesseract.

PaddleOCR does not have a Gujarati language model. This module provides
Tesseract-based OCR specifically for Gujarati (and mixed Gujarati+English)
image-based PDF pages.

Place this file at: app/services/gujarati_ocr.py

Prerequisites:
    apt-get install -y tesseract-ocr tesseract-ocr-guj
    pip install pytesseract

Usage in ocr_extractor.py:
    Replace or wrap ocr_single_page_image() to call this module
    when OCR_LANGUAGE is 'gu'.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Tesseract PSM modes
# 3 = Fully automatic page segmentation (default)
# 6 = Assume uniform block of text
# 11 = Sparse text — find as much text as possible
_PSM_DEFAULT = 3
_PSM_BLOCK = 6
_PSM_SPARSE = 11

# Tesseract OEM modes
# 0 = Legacy engine only
# 3 = Default (Legacy + LSTM)
_OEM_LSTM = 3


def _check_tesseract_available() -> bool:
    """Check if tesseract binary and Gujarati language pack are available."""
    try:
        import pytesseract

        langs = pytesseract.get_languages()
        return "guj" in langs
    except Exception as exc:
        logger.warning("Tesseract not available: %s", exc)
        return False


_TESSERACT_AVAILABLE: Optional[bool] = None


def tesseract_available() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is None:
        _TESSERACT_AVAILABLE = _check_tesseract_available()
    return _TESSERACT_AVAILABLE


def _preprocess_for_gujarati(image: Image.Image) -> np.ndarray:
    """
    Preprocessing specifically tuned for Gujarati script.

    Gujarati has:
    - A horizontal headline (shirorekha) connecting characters
    - Complex matras (vowel diacritics) above and below
    - Conjunct consonants that need clean separation

    Key differences from generic preprocessing:
    - Use a larger adaptive threshold block size (15 vs 11) to handle the
      headline without cutting through character bodies
    - Skip aggressive morphological erosion that breaks matras
    - Use CLAHE for contrast but with smaller tile grid for fine details
    - Upscale to at least 300 DPI equivalent for small text
    """
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    h, w = gray.shape

    # Upscale if image is small — Tesseract accuracy degrades below ~200px height
    min_height = 1200
    if h < min_height:
        scale = min_height / h
        gray = cv2.resize(
            gray, (int(w * scale), min_height), interpolation=cv2.INTER_CUBIC
        )

    # CLAHE with small tile grid to preserve matra detail
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Denoise with median blur (preserves edges better than Gaussian)
    gray = cv2.medianBlur(gray, 3)

    # Adaptive threshold with larger block size for Gujarati headline
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,  # larger than default 11 — handles shirorekha
        C=8,
    )

    # Light dilation to reconnect broken strokes (but not aggressive)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.dilate(binary, kernel, iterations=1)

    return binary


def _run_tesseract(
    image: np.ndarray,
    lang: str = "guj",
    psm: int = _PSM_DEFAULT,
    oem: int = _OEM_LSTM,
    whitelist: Optional[str] = None,
) -> str:
    """Run Tesseract on a preprocessed numpy array and return raw text."""
    import pytesseract

    config = f"--oem {oem} --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    pil_image = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
    return text.strip()


def _compute_text_quality(text: str) -> float:
    """
    Heuristic quality score for Gujarati OCR output.
    Gujarati unicode block: U+0A80–U+0AFF
    """
    if not text or not text.strip():
        return 0.0

    chars = text.replace("\n", "").replace(" ", "")
    if not chars:
        return 0.0

    gujarati_chars = sum(1 for c in chars if "\u0a80" <= c <= "\u0aff")
    latin_chars = sum(1 for c in chars if c.isascii() and c.isalpha())
    digit_chars = sum(1 for c in chars if c.isdigit())
    noise_chars = sum(
        1 for c in chars if not ("\u0a80" <= c <= "\u0aff" or c.isascii())
    )

    total = len(chars)
    gujarati_ratio = gujarati_chars / total
    noise_ratio = noise_chars / total

    # Quality is high if we have Gujarati script or recognisable Latin/digits
    readable = gujarati_chars + latin_chars + digit_chars
    readable_ratio = readable / total if total > 0 else 0.0

    score = min(readable_ratio - noise_ratio * 2, 1.0)
    return max(score, 0.0)


def ocr_gujarati_page(
    image: Image.Image,
    page_number: int,
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    OCR a single page image using Tesseract with Gujarati language.

    Tries multiple PSM modes and picks the best result by text quality.
    Falls back to guj+eng (mixed) if pure guj yields low quality.

    Returns a dict compatible with the existing pipeline page result format.
    """
    started = time.perf_counter()
    warnings: List[str] = []

    if not tesseract_available():
        warnings.append(
            "Tesseract/guj not installed. Run: apt-get install tesseract-ocr tesseract-ocr-guj"
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
        logger.error("Gujarati preprocessing failed page %s: %s", page_number, exc)
        preprocessed = np.array(image.convert("L"))
        warnings.append(f"Preprocessing fallback used: {exc}")

    best_text = ""
    best_score = -1.0

    # Try PSM 3 (auto) first, then PSM 6 (block) for single-column docs
    candidates = [
        ("guj", _PSM_DEFAULT),
        ("guj", _PSM_BLOCK),
        ("guj+eng", _PSM_DEFAULT),  # mixed Gujarati+English (numbers, headings)
    ]

    for lang, psm in candidates:
        try:
            text = _run_tesseract(preprocessed, lang=lang, psm=psm)
            score = _compute_text_quality(text)
            logger.debug(
                "Tesseract page=%s lang=%s psm=%s chars=%s score=%.3f",
                page_number,
                lang,
                psm,
                len(text),
                score,
            )
            if score > best_score:
                best_score = score
                best_text = text
            if score >= 0.6:
                break  # Good enough — don't try further
        except Exception as exc:
            logger.warning(
                "Tesseract failed page=%s lang=%s psm=%s: %s",
                page_number,
                lang,
                psm,
                exc,
            )
            warnings.append(f"Tesseract error (lang={lang} psm={psm}): {exc}")

    if not best_text.strip():
        warnings.append(f"Page {page_number}: Tesseract returned no text.")

    elapsed = time.perf_counter() - started
    logger.info(
        "Gujarati OCR page=%s score=%.3f chars=%s duration=%.2fs",
        page_number,
        best_score,
        len(best_text),
        elapsed,
    )

    return {
        "page_number": page_number,
        "text": best_text,
        "confidence": best_score,
        "warnings": warnings,
        "tables": [],
    }


def extract_gujarati_pdf(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    dpi: int = 200,
    pdf_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    """
    Extract text from all (or selected) pages of a Gujarati image-based PDF.

    Renders each page at the given DPI and runs Tesseract with Gujarati lang.
    Higher DPI (200-300) gives better Gujarati accuracy vs speed tradeoff.

    Args:
        pdf_path: Path to the PDF file
        page_numbers: 1-indexed list of pages to process (None = all)
        dpi: Render resolution — 200 recommended for Gujarati
        pdf_bytes: Optional pre-read bytes to avoid re-reading the file

    Returns:
        List of page result dicts compatible with the extraction pipeline
    """
    import fitz  # PyMuPDF

    try:
        if pdf_bytes:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        else:
            doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {exc}") from exc

    try:
        total = len(doc)
        targets = page_numbers or list(range(1, total + 1))
        targets = [p for p in targets if 1 <= p <= total]

        scale = max(1.5, dpi / 72.0)  # floor at 1.5× to avoid tiny renders
        matrix = fitz.Matrix(scale, scale)
        results: List[Dict[str, Any]] = []

        for page_num in targets:
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            pil_image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if mode == "RGBA":
                pil_image = pil_image.convert("RGB")

            result = ocr_gujarati_page(pil_image, page_number=page_num, dpi=dpi)
            results.append(result)
            logger.debug("Gujarati OCR done page=%s/%s", page_num, total)

    finally:
        doc.close()

    results.sort(key=lambda r: r["page_number"])
    return results
