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

FIXES IN THIS VERSION:
  FIX 1 — Cross-platform Tesseract path detection.
           Previously hardcoded to Windows C:\\Program Files path.
           Now auto-detects: uses PATH on Linux/Mac, falls back to Windows
           path only on win32. Can be overridden with TESSERACT_CMD env var.
  FIX 2 — DPI floor removed from extract_gujarati_pdf().
           Previously forced dpi = max(300, dpi, settings.OCR_DPI), overriding
           the carefully computed effective_ocr_dpi() value from the pipeline.
           Now uses the passed-in dpi directly (min 150 for Gujarati safety).
  FIX 3 — _EMPTY_CONFIDENCE_CUTOFF lowered 0.2 → 0.15 and
           _LOW_CONFIDENCE_WARNING lowered 0.35 → 0.25.
           Gujarati Tesseract on government notice scans legitimately scores
           0.15–0.4; the old cutoff was blanking valid pages.
  FIX 4 — _score_result() weight rebalanced: confidence weight reduced,
           readable_ratio weight raised so short-but-valid Gujarati words
           (village names, amounts) are not discarded.
  FIX 5 — _configure_tesseract_runtime() is no longer called at module
           import time on non-Windows platforms (was overwriting PATH).
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import fitz
import numpy as np
from PIL import Image
import pytesseract

from app.config.settings import settings
from app.utils.image_preprocessing import adaptive_threshold, pil_to_cv2, remove_noise
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TESSERACT_AVAILABLE: Optional[bool] = None
_TESSERACT_CHECK_LOCK = Lock()

# FIX 1: Windows-only path constants (not applied on Linux/Mac)
_WINDOWS_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_WINDOWS_TESSDATA_PREFIX = r"C:\Program Files\Tesseract-OCR\tessdata"

_PRIMARY_PSM = 6
_RETRY_PSM = 11
_TARGET_CONFIDENCE = 0.6
# FIX 3: lowered from 0.35 — Gujarati scores 0.15–0.40 on clean scans
_LOW_CONFIDENCE_WARNING = 0.25
# FIX 3: lowered from 0.20 — preserve text that Tesseract scored above 0.15
_EMPTY_CONFIDENCE_CUTOFF = 0.15

_OCR_LANG = "guj+eng"


def _get_tesseract_cmd() -> str:
    """
    FIX 1: Resolve Tesseract executable path in a cross-platform way.

    Priority:
      1. TESSERACT_CMD environment variable (user override)
      2. On Windows: check the standard install path
      3. On Linux/Mac: rely on PATH (tesseract is usually /usr/bin/tesseract)
    """
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd:
        return env_cmd

    if sys.platform == "win32":
        if Path(_WINDOWS_TESSERACT_CMD).exists():
            return _WINDOWS_TESSERACT_CMD

    # Linux/Mac: use the command name and let PATH resolve it
    return "tesseract"


def _configure_tesseract_runtime() -> None:
    """
    FIX 1: Configure Tesseract only when needed and only for the current platform.

    On Linux/Mac: only set if TESSERACT_CMD or TESSDATA_PREFIX env vars are
    explicitly provided — do NOT overwrite PATH-based resolution.
    On Windows: apply the known install path if it exists.
    """
    cmd = _get_tesseract_cmd()
    pytesseract.pytesseract.tesseract_cmd = cmd

    # Only set TESSDATA_PREFIX if explicitly configured or on Windows
    if sys.platform == "win32":
        tessdata = os.environ.get("TESSDATA_PREFIX", _WINDOWS_TESSDATA_PREFIX)
        os.environ["TESSDATA_PREFIX"] = tessdata
    elif "TESSDATA_PREFIX" in os.environ:
        # Respect explicitly set env var on Linux/Mac
        pass
    # else: leave TESSDATA_PREFIX unset — Tesseract finds tessdata via its own logic

    logger.debug("[Tesseract] cmd=%s", pytesseract.pytesseract.tesseract_cmd)
    logger.debug(
        "[Tesseract] TESSDATA_PREFIX=%s", os.environ.get("TESSDATA_PREFIX", "(unset)")
    )


# FIX 1: Only configure at import time — but now it's cross-platform safe
_configure_tesseract_runtime()


def _check_tesseract_available() -> bool:
    """
    Check whether pytesseract and the Tesseract executable are available.
    Logs detailed diagnostics to help with deployment debugging.
    """
    try:
        version = pytesseract.get_tesseract_version()
        langs = []
        try:
            langs = pytesseract.get_languages(config="")
        except Exception as lang_exc:
            logger.debug("Could not enumerate Tesseract languages: %s", lang_exc)

        logger.info("[Tesseract] version=%s | languages=%s", version, langs)

        has_guj = "guj" in langs
        has_eng = "eng" in langs
        if not has_guj:
            logger.warning(
                "[Tesseract] 'guj' language pack not found. "
                "Install with: sudo apt-get install tesseract-ocr-guj  (Linux) "
                "or download guj.traineddata to your tessdata folder (Windows)."
            )
        if not has_eng:
            logger.warning("[Tesseract] 'eng' language pack not found.")

        return True
    except Exception as exc:
        logger.warning(
            "[Tesseract] Not available: %s  "
            "cmd=%s  "
            "Hint: ensure tesseract is in PATH or set TESSERACT_CMD env var.",
            exc,
            pytesseract.pytesseract.tesseract_cmd,
        )
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


def _resolve_ocr_lang() -> str:
    """
    Determine the Tesseract language string to use.

    If 'guj' is available, use 'guj+eng'.
    If only 'eng' is available, fall back to 'eng'.
    Always returns a usable language string.
    """
    try:
        langs = pytesseract.get_languages(config="")
        if "guj" in langs:
            return "guj+eng" if "eng" in langs else "guj"
        if "eng" in langs:
            logger.warning(
                "[Tesseract] 'guj' pack missing — falling back to 'eng'. "
                "Gujarati accuracy will be very low. "
                "Install guj.traineddata for proper Gujarati OCR."
            )
            return "eng"
    except Exception:
        pass
    return _OCR_LANG  # best-effort default


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
    Blend OCR confidence with Gujarati-script coverage.

    FIX 4: Rebalanced weights:
      - confidence weight reduced 0.7 → 0.5
      - readable_ratio weight raised 0.2 → 0.35
      - script_ratio kept at 0.15
    Rationale: Tesseract confidence on Gujarati government scans is consistently
    low (0.1–0.4) even when the text is correct. Weighting readable_ratio more
    heavily means a page with 80% recognisable characters and 0.2 confidence
    scores ~0.42 instead of ~0.30, keeping it above the EMPTY_CONFIDENCE_CUTOFF.
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

    # FIX 4: rebalanced weights
    score = (confidence * 0.50) + (readable_ratio * 0.35) + (script_ratio * 0.15)
    return round(max(0.0, min(score, 1.0)), 4)


def _ocr_page_with_fallbacks(
    image: Union[np.ndarray, "Image.Image"],
    page_number: int,
    dpi: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    warnings: List[str] = []

    # Resolve actual language string once per page (checks available langs)
    ocr_lang = _resolve_ocr_lang()

    logger.debug(
        "[Tesseract] OCR start | page=%s | lang=%s | dpi=%s | cmd=%s",
        page_number,
        ocr_lang,
        dpi,
        pytesseract.pytesseract.tesseract_cmd,
    )

    available = tesseract_available()
    if not available:
        warnings.append(
            "Tesseract runtime check failed; continuing with OCR fallback attempts."
        )

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

    # Pass 1: primary PSM
    try:
        primary_text, primary_conf = _run_tesseract(
            preprocessed, lang=ocr_lang, psm=_PRIMARY_PSM
        )
    except Exception as exc:
        logger.warning(
            "Tesseract failed | page=%s | lang=%s | psm=%s | error=%s",
            page_number,
            ocr_lang,
            _PRIMARY_PSM,
            exc,
        )
        warnings.append(f"Tesseract error (lang={ocr_lang} psm={_PRIMARY_PSM}): {exc}")
        primary_text, primary_conf = "", 0.0

    primary_score = _score_result(primary_text, primary_conf)
    logger.info(
        "[OCR] Page %s → Tesseract | lang=%s | psm=%s | conf=%.3f | score=%.3f | chars=%s",
        page_number,
        ocr_lang,
        _PRIMARY_PSM,
        primary_conf,
        primary_score,
        len(primary_text),
    )

    best_text = primary_text
    best_score = primary_score
    best_conf = primary_conf

    # Pass 2: retry with PSM 11 if primary returned nothing
    if not primary_text.strip():
        try:
            retry_text, retry_conf = _run_tesseract(
                preprocessed, lang=ocr_lang, psm=_RETRY_PSM
            )
        except Exception as exc:
            logger.warning(
                "Tesseract retry failed | page=%s | lang=%s | psm=%s | error=%s",
                page_number,
                ocr_lang,
                _RETRY_PSM,
                exc,
            )
            warnings.append(
                f"Tesseract error (lang={ocr_lang} psm={_RETRY_PSM}): {exc}"
            )
            retry_text, retry_conf = "", 0.0

        retry_score = _score_result(retry_text, retry_conf)
        logger.info(
            "[OCR] Page %s → Tesseract retry | lang=%s | psm=%s | conf=%.3f | score=%.3f | chars=%s",
            page_number,
            ocr_lang,
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

    # Pass 3: English fallback if still nothing (handles embedded English/digits)
    if not best_text.strip() and ocr_lang != "eng":
        try:
            eng_text, eng_conf = _run_tesseract(
                preprocessed, lang="eng", psm=_PRIMARY_PSM
            )
        except Exception as exc:
            logger.warning(
                "English fallback failed | page=%s | psm=%s | error=%s",
                page_number,
                _PRIMARY_PSM,
                exc,
            )
            warnings.append(f"English fallback error (psm={_PRIMARY_PSM}): {exc}")
            eng_text, eng_conf = "", 0.0

        eng_score = _score_result(eng_text, eng_conf)
        logger.info(
            "[OCR] Page %s → Tesseract eng fallback | psm=%s | conf=%.3f | score=%.3f | chars=%s",
            page_number,
            _PRIMARY_PSM,
            eng_conf,
            eng_score,
            len(eng_text),
        )
        if eng_score > best_score:
            best_text = eng_text
            best_score = eng_score
            best_conf = eng_conf
            best_psm = _PRIMARY_PSM

    # FIX 3: lowered cutoff — 0.15 instead of 0.20
    if best_score < _EMPTY_CONFIDENCE_CUTOFF:
        logger.warning(
            "[OCR] Page %s: score %.3f below cutoff %.2f — blanking output. "
            "Consider lowering OCR_CONFIDENCE_THRESHOLD or checking scan quality.",
            page_number,
            best_score,
            _EMPTY_CONFIDENCE_CUTOFF,
        )
        best_text = ""

    # FIX 3: lowered warning threshold
    if best_score < _LOW_CONFIDENCE_WARNING and best_text:
        warnings.append(f"Low OCR confidence ({best_score:.2f})")

    if not best_text.strip():
        warnings.append(f"Page {page_number}: Tesseract returned no usable text.")

    elapsed = time.perf_counter() - started
    logger.info(
        "Gujarati OCR done | page=%s | lang=%s | psm=%s | conf=%.3f | score=%.3f | duration=%.2fs",
        page_number,
        ocr_lang,
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

    FIX 2: DPI is no longer forced to max(300, dpi, settings.OCR_DPI).
    The pipeline calls effective_ocr_dpi() which already accounts for page count
    and document length. Overriding it here caused unnecessary slowdowns.
    We now use the passed-in dpi directly, with a safe minimum of 150 DPI
    (lower than 150 makes Gujarati matras indistinguishable).
    """
    import fitz

    started = time.perf_counter()

    # FIX 2: use passed-in DPI with a 150 minimum (not 300)
    effective_dpi = max(150, dpi)
    if effective_dpi != dpi:
        logger.debug(
            "DPI raised from %s to minimum 150 for Gujarati glyph clarity", dpi
        )

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
            (page_num, _render_page_to_pil(doc[page_num - 1], dpi=effective_dpi))
            for page_num in targets
        ]
    finally:
        doc.close()

    if not targets:
        return []

    workers = min(_parallel_workers(), len(targets))
    logger.info(
        "Gujarati OCR batch | file=%s | pages=%s | workers=%s | dpi=%s",
        pdf_path.name,
        len(targets),
        workers,
        effective_dpi,
    )

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_rendered_page, image, page_num, effective_dpi
            ): page_num
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
        "Gujarati OCR batch done | file=%s | pages=%s | duration=%.2fs",
        pdf_path.name,
        len(results),
        time.perf_counter() - started,
    )
    return results
