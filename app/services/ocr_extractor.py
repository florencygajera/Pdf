"""
OCR Extraction Engine.

Optimized for Windows + Celery thread pools:
  - converts PDF pages once per batch
  - processes pages concurrently with a bounded thread pool
  - uses a small reusable PaddleOCR pool instead of a single global lock
  - exposes a bytes-based entry point to avoid repeated disk reads
"""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import Any, Dict, List, Optional

import fitz
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError

from app.config.constants import LINE_Y_TOLERANCE_OCR
from app.config.settings import settings
from app.utils.image_preprocessing import preprocess_page_image
from app.utils.logger import get_logger
from app.utils.sorting import merge_hyphenated_lines, sort_ocr_results

logger = get_logger(__name__)

try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass


class _PaddleOCRPool:
    """
    Small bounded pool of PaddleOCR instances.

    PaddleOCR model initialization is expensive. Keeping a tiny reusable pool
    avoids repeated loads while still allowing bounded parallel inference.
    """

    def __init__(self, max_instances: int) -> None:
        self.max_instances = max(1, max_instances)
        self._available: Queue = Queue()
        self._created = 0
        self._create_lock = Lock()
        self._init_error: Optional[Exception] = None

    def _create_instance(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            self._init_error = exc
            raise

        lang = settings.OCR_LANGUAGES[0] if settings.OCR_LANGUAGES else "en"
        instance = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=settings.OCR_USE_GPU,
            show_log=False,
        )
        logger.info(
            "PaddleOCR instance initialized | lang=%s | gpu=%s | slot=%s/%s",
            lang,
            settings.OCR_USE_GPU,
            self._created + 1,
            self.max_instances,
        )
        return instance

    @contextmanager
    def borrow(self):
        if self._init_error is not None:
            raise RuntimeError(
                "PaddleOCR is not available. Install: pip install paddlepaddle paddleocr"
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
                            "PaddleOCR is not available. Install: pip install paddlepaddle paddleocr"
                        ) from exc
                else:
                    instance = self._available.get()

        try:
            yield instance
        finally:
            self._available.put(instance)


_ocr_pool = None
_ocr_pool_lock = Lock()


def _get_ocr_pool() -> _PaddleOCRPool:
    global _ocr_pool
    if _ocr_pool is not None:
        return _ocr_pool

    with _ocr_pool_lock:
        if _ocr_pool is not None:
            return _ocr_pool

        _ocr_pool = _PaddleOCRPool(max_instances=settings.effective_ocr_page_workers)
        return _ocr_pool


def _render_pages_with_fitz(
    *,
    pdf_path: Optional[Path] = None,
    pdf_bytes: Optional[bytes] = None,
    page_numbers: Optional[List[int]] = None,
    dpi: int,
) -> List[Image.Image]:
    """
    Render selected PDF pages directly with PyMuPDF.

    This avoids the Poppler dependency required by pdf2image and is usually
    faster on Windows for OCR workloads.
    """
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


def _filter_by_confidence(ocr_results: List, threshold: float) -> List:
    filtered = []
    for item in ocr_results:
        if not item or len(item) < 2:
            continue
        _, (text, conf) = item
        if conf >= threshold:
            filtered.append(item)
    return filtered


def _ocr_results_to_text(ocr_results: List) -> str:
    if not ocr_results:
        return ""

    sorted_results = sort_ocr_results(ocr_results)

    def top_y(r):
        return min(pt[1] for pt in r[0])

    lines: List[List[str]] = []
    current_line: List[str] = []
    current_y: Optional[float] = None

    for item in sorted_results:
        _, (text, _) = item
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
    if not ocr_results:
        return 0.0
    confs = [item[1][1] for item in ocr_results if item and len(item) >= 2]
    return sum(confs) / len(confs) if confs else 0.0


def _run_paddle_ocr(image_array: np.ndarray) -> List:
    """Run PaddleOCR using a bounded shared pool."""
    pool = _get_ocr_pool()

    try:
        with pool.borrow() as ocr:
            result = ocr.ocr(image_array, cls=True)
        if result and isinstance(result[0], list):
            return result[0]
        return result or []
    except Exception as exc:
        logger.error("PaddleOCR inference failed: %s", exc, exc_info=True)
        return []


def ocr_single_page_image(image, page_number: int) -> Dict[str, Any]:
    """
    Run OCR on a single PIL image.

    The expensive model is shared through the pool, while preprocessing stays
    per-page and thread-safe.
    """
    warnings: List[str] = []
    rendered_image = np.array(image)

    try:
        processed_arr, preprocess_meta = preprocess_page_image(image)
    except Exception as exc:
        logger.error(
            "Preprocessing failed on page %s: %s", page_number, exc, exc_info=True
        )
        warnings.append(f"Preprocessing failed: {exc}")
        import cv2

        processed_arr = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
        preprocess_meta = {}

    if preprocess_meta.get("deskew_angle", 0) != 0:
        warnings.append(f"Page was deskewed by {preprocess_meta['deskew_angle']:.1f}°")

    try:
        raw_results = _run_paddle_ocr(processed_arr)
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
            "rendered_image": rendered_image,
        }

    filtered = _filter_by_confidence(raw_results, settings.OCR_CONFIDENCE_THRESHOLD)
    confidence = _compute_page_confidence(filtered)
    text = _ocr_results_to_text(filtered)

    return {
        "page_number": page_number,
        "text": text,
        "confidence": confidence,
        "warnings": warnings,
        "raw_results": filtered,
        "rendered_image": rendered_image,
    }


def _process_images_in_threads(
    images: List,
    start_page: int,
    target_set: Optional[set],
) -> List[Dict[str, Any]]:
    """Shared page-processing helper for both path- and bytes-based inputs."""
    results: List[Dict[str, Any]] = []
    max_workers = min(settings.effective_ocr_page_workers, len(images)) if len(images) > 1 else 1

    def _process(item):
        idx, pil_image = item
        page_num = start_page + idx
        if target_set and page_num not in target_set:
            return None
        return ocr_single_page_image(pil_image, page_num)

    if max_workers == 1:
        for item in enumerate(images):
            result = _process(item)
            if result:
                results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process, item) for item in enumerate(images)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        results.sort(key=lambda item: item["page_number"])

    gc.collect()
    return results


def extract_ocr_pdf(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    dpi: int = None,
) -> List[Dict[str, Any]]:
    """Full OCR extraction pipeline using the PDF on disk."""
    dpi = dpi or settings.OCR_DPI

    try:
        all_images = _render_pages_with_fitz(
            pdf_path=pdf_path,
            page_numbers=page_numbers,
            dpi=dpi,
        )
    except Exception as fitz_exc:
        try:
            all_images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt="PNG",
                thread_count=settings.effective_ocr_pdf2image_threads,
                first_page=min(page_numbers) if page_numbers else None,
                last_page=max(page_numbers) if page_numbers else None,
            )
        except PDFSyntaxError as exc:
            raise ValueError(f"PDF syntax error during image conversion: {exc}") from exc
        except PDFPageCountError as exc:
            raise ValueError(f"Cannot count PDF pages: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"pdf rendering failed: {fitz_exc}; pdf2image: {exc}") from exc

    target_set = set(page_numbers) if page_numbers else None
    start_page = min(page_numbers) if page_numbers else 1
    results = _process_images_in_threads(all_images, start_page, target_set)

    logger.info(
        "OCR extraction complete | pages=%s | file=%s",
        len(results),
        pdf_path.name,
    )
    return results


def extract_ocr_pdf_from_bytes(
    pdf_bytes: bytes,
    page_numbers: Optional[List[int]] = None,
    dpi: int = None,
) -> List[Dict[str, Any]]:
    """Bytes-based OCR pipeline to avoid repeated disk reads."""
    dpi = dpi or settings.OCR_DPI

    try:
        all_images = _render_pages_with_fitz(
            pdf_bytes=pdf_bytes,
            page_numbers=page_numbers,
            dpi=dpi,
        )
    except Exception as fitz_exc:
        try:
            all_images = convert_from_bytes(
                pdf_bytes,
                dpi=dpi,
                fmt="PNG",
                thread_count=settings.effective_ocr_pdf2image_threads,
                first_page=min(page_numbers) if page_numbers else None,
                last_page=max(page_numbers) if page_numbers else None,
            )
        except PDFSyntaxError as exc:
            raise ValueError(f"PDF syntax error during image conversion: {exc}") from exc
        except PDFPageCountError as exc:
            raise ValueError(f"Cannot count PDF pages: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"pdf rendering failed: {fitz_exc}; pdf2image: {exc}") from exc

    target_set = set(page_numbers) if page_numbers else None
    start_page = min(page_numbers) if page_numbers else 1
    results = _process_images_in_threads(all_images, start_page, target_set)

    logger.info("OCR extraction complete | pages=%s | bytes input", len(results))
    return results
