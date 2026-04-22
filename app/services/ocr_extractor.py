"""
OCR Extraction Engine.

FIXES IN THIS VERSION:

  FIX A — strict routing: Gujarati OCR goes to app.services.gujarati_ocr
           only, English OCR goes to PaddleOCR only.
  FIX B — PaddleOCR is pinned to settings.PADDLE_LANG so it never receives
           lang="gu".
  FIX C — rendered pages are generated at a minimum of 300 DPI and kept as
           PIL images before OCR.
  FIX D — _ocr_result_looks_good() confidence gate lowered 0.4 → 0.25.
  FIX E — shutdown_ocr_executor() is available for clean FastAPI shutdown.
  FIX F — _get_multiprocessing_context() exported for tests.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import Any, Dict, List, Optional

import fitz
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError

from app.config.constants import line_y_tolerance_ocr
from app.config.settings import settings
from app.services.gujarati_ocr import extract_gujarati_pdf, ocr_gujarati_page
from app.utils.image_preprocessing import (
    estimate_page_complexity,
    preprocess_page_image,
)
from app.utils.logger import get_logger
from app.utils.sorting import merge_hyphenated_lines, sort_ocr_results

logger = get_logger(__name__)

try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass


def _get_multiprocessing_context():
    """
    Returns the appropriate multiprocessing context for the platform.
    Uses 'fork' on Linux (faster worker startup), 'spawn' on Windows/macOS.

    FIX E: Exported for testability — tests monkeypatch sys.platform, os.name,
    and mp.get_context to verify the correct context is chosen on each platform.
    """
    if sys.platform.startswith("linux") and os.name == "posix":
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _normalize_ocr_language(language: Optional[str]) -> str:
    value = (language or "").strip().lower()
    if value in {"gu", "guj", "gujarati"}:
        return "gujarati"
    if value in {"en", "eng", "english"}:
        return "english"
    return value


def _resolve_ocr_language(language: Optional[str] = None) -> str:
    """
    Resolve the OCR language routed by the pipeline.

    Strict routing:
      - Gujarati => Tesseract-only via app.services.gujarati_ocr
      - English  => PaddleOCR-only
    """
    normalized = _normalize_ocr_language(language)
    if normalized in {"gujarati", "english"}:
        return normalized

    engine = (settings.OCR_ENGINE or "hybrid").strip().lower()
    if engine == "tesseract":
        return "gujarati"
    if engine == "paddle":
        return "english"
    return "english"


class _PaddleOCRPool:
    """
    Small bounded pool of PaddleOCR instances.
    PaddleOCR model initialisation is expensive (~5–10 s per instance).
    """

    def __init__(self, max_instances: int) -> None:
        self.max_instances = max(1, max_instances)
        self._available: Queue = Queue()
        self._created = 0
        self._create_lock = Lock()
        self._init_error: Optional[Exception] = None

    def _create_instance(self):
        try:
            _configure_paddle_runtime()
            from paddleocr import PaddleOCR
        except ImportError as exc:
            self._init_error = exc
            raise

        primary_lang = (settings.PADDLE_LANG or "en").strip() or "en"

        instance = PaddleOCR(
            use_angle_cls=True,
            lang=primary_lang,
            use_gpu=settings.OCR_USE_GPU,
            show_log=False,
        )
        logger.info(
            "PaddleOCR instance initialised | lang=%s | gpu=%s | slot=%s/%s",
            primary_lang,
            settings.OCR_USE_GPU,
            self._created + 1,
            self.max_instances,
        )
        return instance

    @contextmanager
    def borrow(self):
        if self._init_error is not None:
            raise RuntimeError(
                "PaddleOCR is not available. "
                "Install: pip install paddlepaddle paddleocr"
            ) from self._init_error

        try:
            instance = self._available.get_nowait()
        except Empty:
            instance = None

        if instance is None:
            with self._create_lock:
                if self._created < self.max_instances:
                    try:
                        instance = self._create_instance()
                        self._created += 1
                    except ImportError as exc:
                        raise RuntimeError(
                            "PaddleOCR is not available. "
                            "Install: pip install paddlepaddle paddleocr"
                        ) from exc
                else:
                    try:
                        instance = self._available.get(timeout=60)
                    except Empty:
                        logger.warning(
                            "OCR pool stalled for 60 s; creating overflow instance "
                            "(pool max=%s). Check for hung OCR workers.",
                            self.max_instances,
                        )
                        instance = self._create_instance()

        try:
            yield instance
        finally:
            self._available.put(instance)


_ocr_pool: Optional[_PaddleOCRPool] = None
_ocr_pool_lock = Lock()
_paddle_runtime_configured = False
_paddle_runtime_lock = Lock()

# FIX D: Thread-safe warning flag — plain bool caused a race condition when
# multiple page threads hit the same PaddleOCR error simultaneously, producing
# duplicate warning log spam.
_ocr_runtime_warning_emitted = False
_ocr_runtime_warning_lock = threading.Lock()

_ocr_executor = None
_ocr_executor_lock = Lock()


def _normalize_page_numbers(
    page_numbers: Optional[List[int]], total_pages: int
) -> List[int]:
    if page_numbers:
        return [p for p in page_numbers if 1 <= p <= total_pages]
    return list(range(1, total_pages + 1))


def _chunk_list(lst: List, size: int) -> List[List]:
    if size <= 0:
        return [lst]
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _get_ocr_pool() -> _PaddleOCRPool:
    global _ocr_pool
    if _ocr_pool is not None:
        return _ocr_pool
    with _ocr_pool_lock:
        if _ocr_pool is not None:
            return _ocr_pool
        _ocr_pool = _PaddleOCRPool(max_instances=settings.effective_ocr_page_workers)
        return _ocr_pool


def _ocr_worker_initializer() -> None:
    """Initialise Paddle runtime and preload the shared OCR pool in each worker."""
    _configure_paddle_runtime()
    pool = _get_ocr_pool()
    try:
        with pool.borrow():
            pass
    except Exception as exc:
        logger.debug("OCR worker warm-up skipped: %s", exc)


def _get_ocr_executor():
    """Return a long-lived OCR process pool reused across requests."""
    global _ocr_executor
    if _ocr_executor is not None:
        return _ocr_executor
    with _ocr_executor_lock:
        if _ocr_executor is not None:
            return _ocr_executor
        ctx = _get_multiprocessing_context()
        _ocr_executor = ProcessPoolExecutor(
            max_workers=max(1, settings.effective_ocr_chunk_workers),
            mp_context=ctx,
            initializer=_ocr_worker_initializer,
        )
        return _ocr_executor


def shutdown_ocr_executor(wait: bool = True) -> None:
    """
    FIX C: Cleanly shut down the OCR ProcessPoolExecutor.
    Called from the FastAPI lifespan shutdown handler to avoid zombie processes.
    """
    global _ocr_executor
    with _ocr_executor_lock:
        if _ocr_executor is not None:
            try:
                _ocr_executor.shutdown(wait=wait, cancel_futures=not wait)
                logger.info("OCR ProcessPoolExecutor shut down (wait=%s).", wait)
            except Exception as exc:
                logger.warning("OCR executor shutdown error: %s", exc)
            finally:
                _ocr_executor = None


def _configure_paddle_runtime() -> None:
    """Disable unstable Paddle runtime accelerators when they are known to crash."""
    global _paddle_runtime_configured
    if _paddle_runtime_configured:
        return
    with _paddle_runtime_lock:
        if _paddle_runtime_configured:
            return
        try:
            import paddle

            paddle.set_flags({"FLAGS_use_mkldnn": bool(settings.OCR_ENABLE_MKLDNN)})
        except Exception as exc:
            logger.debug("Paddle runtime flag config skipped: %s", exc)
        _paddle_runtime_configured = True


def _render_pages_with_fitz(
    *,
    pdf_path: Optional[Path] = None,
    pdf_bytes: Optional[bytes] = None,
    page_numbers: Optional[List[int]] = None,
    dpi: int,
) -> List[Image.Image]:
    """Render PDF pages with PyMuPDF — faster than pdf2image for most PDFs."""
    dpi = max(300, dpi)
    if pdf_bytes is not None:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    elif pdf_path is not None:
        doc = fitz.open(str(pdf_path))
    else:
        raise ValueError("pdf_path or pdf_bytes is required")

    try:
        pages = page_numbers or list(range(1, len(doc) + 1))
        scale = max(1.0, dpi / 72.0)
        matrix = fitz.Matrix(scale, scale)
        images: List[Image.Image] = []
        for page_num in pages:
            if page_num < 1 or page_num > len(doc):
                continue
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if mode == "RGBA":
                image = image.convert("RGB")
            images.append(image)
        return images
    finally:
        doc.close()


def _render_pages_for_ocr(
    *,
    pdf_path: Optional[Path] = None,
    pdf_bytes: Optional[bytes] = None,
    page_numbers: Optional[List[int]] = None,
    dpi: int,
) -> List[Image.Image]:
    """Render pages with PyMuPDF first, falling back to pdf2image if needed."""
    dpi = max(300, dpi)
    try:
        return _render_pages_with_fitz(
            pdf_path=pdf_path,
            pdf_bytes=pdf_bytes,
            page_numbers=page_numbers,
            dpi=dpi,
        )
    except Exception as fitz_exc:
        try:
            first_page = min(page_numbers) if page_numbers else None
            last_page = max(page_numbers) if page_numbers else None
            if pdf_bytes is not None:
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=dpi,
                    fmt="PNG",
                    thread_count=settings.effective_ocr_pdf2image_threads,
                    first_page=first_page,
                    last_page=last_page,
                )
            elif pdf_path is not None:
                images = convert_from_path(
                    str(pdf_path),
                    dpi=dpi,
                    fmt="PNG",
                    thread_count=settings.effective_ocr_pdf2image_threads,
                    first_page=first_page,
                    last_page=last_page,
                )
            else:
                raise ValueError("pdf_path or pdf_bytes is required")

            if page_numbers and first_page is not None and last_page is not None:
                page_to_image = {
                    page_num: image
                    for page_num, image in zip(range(first_page, last_page + 1), images)
                }
                images = [page_to_image[p] for p in page_numbers if p in page_to_image]
            return images
        except PDFSyntaxError as exc:
            raise ValueError(
                f"PDF syntax error during image conversion: {exc}"
            ) from exc
        except PDFPageCountError as exc:
            raise ValueError(f"Cannot count PDF pages: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(
                f"PDF rendering failed: fitz={fitz_exc}; pdf2image={exc}"
            ) from exc


def _filter_by_confidence(ocr_results: List, threshold: float) -> List:
    filtered = []
    for item in ocr_results:
        if not item or len(item) < 2:
            continue
        _, (text, conf) = item
        if conf >= threshold:
            filtered.append(item)
    return filtered


def _ocr_results_to_text(ocr_results: List, dpi: int) -> str:
    if not ocr_results:
        return ""

    y_tolerance = line_y_tolerance_ocr(dpi)
    sorted_results = sort_ocr_results(ocr_results, y_tolerance=y_tolerance)

    def top_y(r):
        return min(pt[1] for pt in r[0])

    lines: List[List[str]] = []
    current_line: List[str] = []
    current_y: Optional[float] = None

    for item in sorted_results:
        _, (text, _) = item
        y = top_y(item)
        if current_y is None or abs(y - current_y) > y_tolerance:
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
    if not ocr_results:
        return 0.0
    confs = [item[1][1] for item in ocr_results if item and len(item) >= 2]
    return sum(confs) / len(confs) if confs else 0.0


def _run_paddle_ocr(image_array: np.ndarray) -> List:
    """Run PaddleOCR using a bounded shared pool."""
    global _ocr_runtime_warning_emitted
    pool = _get_ocr_pool()

    try:
        with pool.borrow() as ocr:
            result = ocr.ocr(image_array, cls=True)
        if result and isinstance(result[0], list):
            return result[0]
        return result or []
    except Exception as exc:
        message = str(exc)
        if (
            "OneDnnContext does not have the input Filter" in message
            or "fused_conv2d" in message
        ):
            # FIX D: thread-safe warning flag to avoid duplicate log spam
            with _ocr_runtime_warning_lock:
                if not _ocr_runtime_warning_emitted:
                    logger.warning(
                        "PaddleOCR runtime issue detected; returning empty result: %s",
                        message,
                    )
                    _ocr_runtime_warning_emitted = True
        else:
            logger.error("PaddleOCR inference failed: %s", exc)
        return []


def _page_array_from_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array."""
    return np.asarray(image)


def _build_ocr_page_result(
    page_number: int,
    warnings: List[str],
    raw_results: List,
    include_rendered_image: bool,
    rendered_image: Optional[np.ndarray],
    render_dpi: int,
) -> Dict[str, Any]:
    filtered = _filter_by_confidence(raw_results, settings.OCR_CONFIDENCE_THRESHOLD)
    confidence = _compute_page_confidence(filtered)
    text = _ocr_results_to_text(filtered, dpi=render_dpi)
    return {
        "page_number": page_number,
        "text": text,
        "confidence": confidence,
        "warnings": warnings,
        "raw_results": filtered,
        "rendered_image": rendered_image if include_rendered_image else None,
    }


def _ocr_result_looks_good(page_result: Dict[str, Any]) -> bool:
    """
    Fast-path acceptance gate for OCR results.

    FIX B: Confidence threshold lowered from 0.4 → 0.25 and minimum text
    length lowered from 15 → 8 characters.

    Reasoning:
    - PaddleOCR's Gujarati/Indic model assigns lower confidence scores than
      the English model for structurally valid characters.
    - Government notice PDFs in Gujarati contain short fields like registration
      numbers, village names, and amounts that are 5–10 characters long.
    - At the old thresholds (0.4 / 15 chars) virtually all valid Gujarati fast-
      path results were rejected, resulting in empty text output.
    """
    text = (page_result.get("text") or "").strip()
    if not text:
        return False
    confidence = float(page_result.get("confidence") or 0.0)
    # FIX B: was 0.4 — Indic model scores valid words at 0.25–0.6
    if confidence < 0.25:
        return False
    # FIX B: was 15 — Gujarati words like "GANGAD" or "161600" are 6 chars
    return len(text) >= 8 or len(text.split()) >= 2


def _ocr_chunk_worker_from_path(payload) -> List[Dict[str, Any]]:
    """OCR a chunk in a separate process using the PDF path as source."""
    path_str, mtime_ns, size, page_numbers, dpi = payload
    return extract_ocr_pdf_local(Path(path_str), page_numbers=page_numbers, dpi=dpi)


def ocr_single_page_image(
    image,
    page_number: int,
    include_rendered_image: bool = True,
    dpi: Optional[int] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run OCR on a single PIL image.

    Strict routing:
      - Gujarati -> Tesseract only
      - English  -> PaddleOCR only

    PaddleOCR path:
      1. Fast path — run PaddleOCR on the raw rendered image. Accept if
         the result looks good (confidence ≥ 0.25, text ≥ 8 chars).
      2. Slow path — apply full image preprocessing (deskew, denoise,
         adaptive threshold, morphological cleanup) then re-run OCR.
      3. Return whichever path gave the higher combined score.
    """
    warnings: List[str] = []
    render_dpi = max(300, dpi or settings.OCR_DPI)
    resolved_language = _resolve_ocr_language(language)

    if resolved_language == "gujarati":
        logger.info("[OCR] Page %s -> Engine: Tesseract", page_number)
        result = ocr_gujarati_page(image, page_number=page_number, dpi=render_dpi)
        if include_rendered_image and "rendered_image" not in result:
            result["rendered_image"] = _page_array_from_image(image)
        return result

    logger.info("[OCR] Page %s -> Engine: Paddle", page_number)
    raw_image = _page_array_from_image(image)
    rendered_image = raw_image if include_rendered_image else None
    profile = estimate_page_complexity(raw_image)

    fast_page = None
    try:
        fast_arr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        fast_results = _run_paddle_ocr(fast_arr)
        if fast_results:
            fast_page = _build_ocr_page_result(
                page_number=page_number,
                warnings=warnings[:],
                raw_results=fast_results,
                include_rendered_image=include_rendered_image,
                rendered_image=rendered_image,
                render_dpi=render_dpi,
            )
            if _ocr_result_looks_good(fast_page):
                return fast_page
    except Exception as exc:
        logger.debug("Fast OCR path failed on page %s: %s", page_number, exc)
        fast_page = None

    # Slow path — full preprocessing
    try:
        processed_arr, preprocess_meta = preprocess_page_image(
            image,
            prefer_light=not profile["needs_full_preprocess"],
            apply_deskew=profile["edge_ratio"] > 0.02,
        )
    except Exception as exc:
        logger.error(
            "Preprocessing failed on page %s: %s", page_number, exc, exc_info=True
        )
        warnings.append(f"Preprocessing failed: {exc}")
        processed_arr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        preprocess_meta = {}

    if preprocess_meta.get("deskew_angle", 0) != 0:
        warnings.append(f"Page deskewed by {preprocess_meta['deskew_angle']:.1f}°")

    try:
        heavy_results = _run_paddle_ocr(processed_arr)
    except Exception as exc:
        logger.error("OCR failed on page %s: %s", page_number, exc, exc_info=True)
        return {
            "page_number": page_number,
            "text": "",
            "confidence": 0.0,
            "warnings": [f"OCR engine error: {exc}"],
            "raw_results": [],
            "rendered_image": rendered_image,
        }

    if not heavy_results:
        warnings.append(
            f"Page {page_number}: OCR returned no results (blank or unreadable)."
        )
        return {
            "page_number": page_number,
            "text": "",
            "confidence": 0.0,
            "warnings": warnings,
            "raw_results": [],
            "rendered_image": rendered_image,
        }

    heavy_page = _build_ocr_page_result(
        page_number=page_number,
        warnings=warnings,
        raw_results=heavy_results,
        include_rendered_image=include_rendered_image,
        rendered_image=rendered_image,
        render_dpi=render_dpi,
    )

    # Return the better result between fast and heavy paths
    try:
        fast_score = (
            len(fast_page.get("text", ""))
            + float(fast_page.get("confidence", 0.0)) * 100.0
            if fast_page
            else -1
        )
    except Exception:
        fast_score = -1

    heavy_score = (
        len(heavy_page.get("text", ""))
        + float(heavy_page.get("confidence", 0.0)) * 100.0
    )

    if fast_score >= heavy_score and fast_page and _ocr_result_looks_good(fast_page):
        return fast_page
    return heavy_page


def _process_images_in_threads(
    images: List,
    start_page: int,
    page_numbers: Optional[List[int]],
    dpi: int,
    language: Optional[str] = None,
    parallel: bool = True,
) -> List[Dict[str, Any]]:
    """Process page images through OCR using a thread pool."""
    results: List[Dict[str, Any]] = []
    max_workers = (
        min(settings.effective_ocr_page_workers, len(images)) if len(images) > 1 else 1
    )

    def _process(item):
        idx, pil_image = item
        page_num = page_numbers[idx] if page_numbers else start_page + idx
        return ocr_single_page_image(pil_image, page_num, dpi=dpi, language=language)

    if not parallel or max_workers == 1:
        for item in enumerate(images):
            results.append(_process(item))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process, item): item for item in enumerate(images)
            }
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda item: item["page_number"])
    gc.collect()
    return results


def _extract_scanned_tables_from_page(
    rendered_image: np.ndarray,
    raw_results: List,
    page_number: int,
) -> List[Dict[str, Any]]:
    if not raw_results:
        return []
    try:
        from app.services.table_extractor import extract_tables_scanned
    except Exception as exc:
        logger.warning(
            "Scanned table extractor unavailable for page %s: %s", page_number, exc
        )
        return []
    try:
        return extract_tables_scanned(rendered_image, raw_results, page_number)
    except Exception as exc:
        logger.warning("Scanned table extraction failed page %s: %s", page_number, exc)
        return []


def extract_ocr_pdf(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    dpi: int = None,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Full OCR extraction pipeline using the PDF on disk."""
    dpi = max(300, dpi or settings.OCR_DPI)
    resolved_language = _resolve_ocr_language(language)
    if resolved_language == "gujarati":
        logger.info(
            "[OCR] Routing scanned pages to Tesseract | pages=%s | dpi=%s",
            page_numbers if page_numbers else "all",
            dpi,
        )
        return extract_gujarati_pdf(
            pdf_path,
            page_numbers=page_numbers,
            dpi=dpi,
        )

    try:
        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
    except Exception:
        total_pages = 0

    normalized_pages = (
        _normalize_page_numbers(page_numbers, total_pages)
        if total_pages
        else (page_numbers or [])
    )
    if not normalized_pages and total_pages:
        normalized_pages = list(range(1, total_pages + 1))

    chunk_size = settings.effective_ocr_chunk_size(len(normalized_pages) or 1)
    chunks = _chunk_list(normalized_pages, chunk_size) if normalized_pages else []
    use_process_pool = (
        settings.OCR_PARALLEL_INFERENCE
        and len(chunks) > 1
        and settings.effective_ocr_chunk_workers > 1
    )

    if use_process_pool:
        logger.info(
            "OCR using shared process pool | pages=%s | chunks=%s | workers=%s | dpi=%s",
            len(normalized_pages),
            len(chunks),
            settings.effective_ocr_chunk_workers,
            dpi,
        )
        try:
            stat = pdf_path.stat()
            executor = _get_ocr_executor()
            futures = [
                executor.submit(
                    _ocr_chunk_worker_from_path,
                    (str(pdf_path), stat.st_mtime_ns, stat.st_size, chunk, dpi),
                )
                for chunk in chunks
            ]
            results = []
            for future in as_completed(futures):
                results.extend(future.result())
        except Exception as exc:
            logger.warning(
                "Process pool OCR failed, falling back to sequential: %s", exc
            )
            all_images = _render_pages_for_ocr(
                pdf_path=pdf_path, page_numbers=normalized_pages, dpi=dpi
            )
            results = _process_images_in_threads(
                all_images,
                start_page=normalized_pages[0] if normalized_pages else 1,
                page_numbers=normalized_pages,
                dpi=dpi,
                language=resolved_language,
                parallel=False,
            )
    else:
        all_images = _render_pages_for_ocr(
            pdf_path=pdf_path, page_numbers=normalized_pages, dpi=dpi
        )
        results = _process_images_in_threads(
            all_images,
            start_page=normalized_pages[0] if normalized_pages else 1,
            page_numbers=normalized_pages,
            dpi=dpi,
            language=resolved_language,
            parallel=False,
        )

    results.sort(key=lambda item: item.get("page_number", 0))
    logger.info(
        "OCR extraction complete | pages=%s | file=%s", len(results), pdf_path.name
    )
    return results


def extract_ocr_pdf_from_bytes(
    pdf_bytes: bytes,
    page_numbers: Optional[List[int]] = None,
    dpi: int = None,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Bytes-based OCR pipeline to avoid repeated disk reads."""
    dpi = max(300, dpi or settings.OCR_DPI)
    resolved_language = _resolve_ocr_language(language)
    if resolved_language == "gujarati":
        logger.info(
            "[OCR] Routing scanned pages to Tesseract | pages=%s | dpi=%s",
            page_numbers if page_numbers else "all",
            dpi,
        )
        return extract_gujarati_pdf(
            Path("memory.pdf"),
            page_numbers=page_numbers,
            dpi=dpi,
            pdf_bytes=pdf_bytes,
        )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()
    except Exception:
        total_pages = 0

    normalized_pages = (
        _normalize_page_numbers(page_numbers, total_pages)
        if total_pages
        else (page_numbers or [])
    )
    if not normalized_pages and total_pages:
        normalized_pages = list(range(1, total_pages + 1))

    all_images = _render_pages_for_ocr(
        pdf_bytes=pdf_bytes, page_numbers=normalized_pages, dpi=dpi
    )
    results = _process_images_in_threads(
        all_images,
        start_page=normalized_pages[0] if normalized_pages else 1,
        page_numbers=normalized_pages,
        dpi=dpi,
        language=resolved_language,
        parallel=len(all_images) > 1,
    )

    logger.info("OCR extraction complete | pages=%s | bytes input", len(results))
    return results


def extract_ocr_pdf_local(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    dpi: int = None,
) -> List[Dict[str, Any]]:
    """
    Local OCR path for process-pool chunk workers.
    Runs rendering + OCR sequentially to avoid nested process pools.
    """
    dpi = max(300, dpi or settings.OCR_DPI)
    all_images = _render_pages_for_ocr(
        pdf_path=pdf_path, page_numbers=page_numbers, dpi=dpi
    )
    start_page = min(page_numbers) if page_numbers else 1
    results: List[Dict[str, Any]] = []

    for idx, pil_image in enumerate(all_images):
        page_num = page_numbers[idx] if page_numbers else start_page + idx
        page_info = ocr_single_page_image(
            pil_image,
            page_num,
            include_rendered_image=False,
            dpi=dpi,
            language="english",
        )

        if page_info.get("text") or page_info.get("raw_results"):
            rendered_image = _page_array_from_image(pil_image)
            page_info["tables"] = _extract_scanned_tables_from_page(
                rendered_image, page_info.get("raw_results", []), page_num
            )
        else:
            page_info["tables"] = []

        page_info.pop("rendered_image", None)
        results.append(page_info)

    results.sort(key=lambda item: item["page_number"])
    return results
