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
import os
import multiprocessing as mp
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from io import BytesIO
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
            _configure_paddle_runtime()
            from paddleocr import PaddleOCR
        except ImportError as exc:
            self._init_error = exc
            raise

        lang = ",".join(settings.OCR_LANGUAGES) if settings.OCR_LANGUAGES else "en"
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
                    try:
                        instance = self._available.get(timeout=60)
                    except Empty as exc:
                        raise RuntimeError(
                            "OCR pool deadlocked: no instance returned within 60s. "
                            "A worker may have crashed mid-inference."
                        ) from exc

        try:
            yield instance
        finally:
            self._available.put(instance)


_ocr_pool = None
_ocr_pool_lock = Lock()
_paddle_runtime_configured = False
_paddle_runtime_lock = Lock()
_ocr_runtime_warning_emitted = False


def _normalize_page_numbers(page_numbers: Optional[List[int]], total_pages: int) -> List[int]:
    if page_numbers:
        return [p for p in page_numbers if 1 <= p <= total_pages]
    return list(range(1, total_pages + 1))


def _chunk_list(lst: List, size: int) -> List[List]:
    if size <= 0:
        return [lst]
    return [lst[i : i + size] for i in range(0, len(lst), size)]


@lru_cache(maxsize=8)
def _load_pdf_bytes_cached(path_str: str, mtime_ns: int, size: int) -> bytes:
    """Load PDF bytes once per worker process for path-based OCR chunks."""
    with open(path_str, "rb") as f:
        return f.read()


def _get_ocr_pool() -> _PaddleOCRPool:
    global _ocr_pool
    if _ocr_pool is not None:
        return _ocr_pool

    with _ocr_pool_lock:
        if _ocr_pool is not None:
            return _ocr_pool

        _ocr_pool = _PaddleOCRPool(max_instances=settings.effective_ocr_page_workers)
        return _ocr_pool


def _configure_paddle_runtime() -> None:
    """
    Disable unstable Paddle runtime accelerators when they are known to crash.

    Some environments emit oneDNN/MKLDNN fused-conv errors during OCR. We
    prefer stable throughput over a faster-but-crashy runtime.
    """
    global _paddle_runtime_configured
    if _paddle_runtime_configured:
        return

    with _paddle_runtime_lock:
        if _paddle_runtime_configured:
            return

        try:
            import paddle

            paddle.set_flags(
                {
                    "FLAGS_use_mkldnn": bool(settings.OCR_ENABLE_MKLDNN),
                }
            )
        except Exception as exc:
            logger.debug("Paddle runtime flag configuration skipped: %s", exc)

        _paddle_runtime_configured = True


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


def _render_pages_for_ocr(
    *,
    pdf_path: Optional[Path] = None,
    pdf_bytes: Optional[bytes] = None,
    page_numbers: Optional[List[int]] = None,
    dpi: int,
) -> List[Image.Image]:
    """Render pages with PyMuPDF first, falling back to pdf2image if needed."""
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
            raise ValueError(f"PDF syntax error during image conversion: {exc}") from exc
        except PDFPageCountError as exc:
            raise ValueError(f"Cannot count PDF pages: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"pdf rendering failed: {fitz_exc}; pdf2image: {exc}") from exc


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
        message = str(exc)
        global _ocr_runtime_warning_emitted
        if (
            "OneDnnContext does not have the input Filter" in message
            or "fused_conv2d" in message
        ):
            if not _ocr_runtime_warning_emitted:
                logger.warning(
                    "PaddleOCR runtime issue detected; returning empty OCR result and keeping pipeline alive: %s",
                    message,
                )
                _ocr_runtime_warning_emitted = True
        else:
            logger.error("PaddleOCR inference failed: %s", exc)
        return []


def _page_array_from_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array only when needed."""
    return np.asarray(image)


def _build_ocr_page_result(
    page_number: int,
    warnings: List[str],
    raw_results: List,
    include_rendered_image: bool,
    rendered_image: Optional[np.ndarray],
) -> Dict[str, Any]:
    filtered = _filter_by_confidence(raw_results, settings.OCR_CONFIDENCE_THRESHOLD)
    confidence = _compute_page_confidence(filtered)
    text = _ocr_results_to_text(filtered)
    return {
        "page_number": page_number,
        "text": text,
        "confidence": confidence,
        "warnings": warnings,
        "raw_results": filtered,
        "rendered_image": rendered_image if include_rendered_image else None,
    }


def _ocr_result_looks_good(page_result: Dict[str, Any]) -> bool:
    text = (page_result.get("text") or "").strip()
    if not text:
        return False
    confidence = float(page_result.get("confidence") or 0.0)
    if confidence < settings.OCR_CONFIDENCE_THRESHOLD:
        return False
    return len(text) >= 20 or len(text.split()) >= 3


def _ocr_chunk_worker(payload) -> List[Dict[str, Any]]:
    """
    OCR a chunk inside a separate process.

    Each process renders its own pages and processes them sequentially. The
    model pool is still reused within the process, but we avoid cross-thread
    sharing entirely.
    """
    pdf_bytes, page_numbers, dpi = payload
    images = _render_pages_for_ocr(
        pdf_bytes=pdf_bytes,
        page_numbers=page_numbers,
        dpi=dpi,
    )
    start_page = page_numbers[0] if page_numbers else 1
    return _process_images_in_threads(
        images,
        start_page=start_page,
        page_numbers=page_numbers,
        parallel=False,
    )


def _ocr_chunk_worker_from_path(payload) -> List[Dict[str, Any]]:
    """OCR a chunk in a separate process using the PDF path as the source."""
    path_str, mtime_ns, size, page_numbers, dpi = payload
    pdf_bytes = _load_pdf_bytes_cached(path_str, mtime_ns, size)
    images = _render_pages_for_ocr(
        pdf_bytes=pdf_bytes,
        page_numbers=page_numbers,
        dpi=dpi,
    )
    start_page = page_numbers[0] if page_numbers else 1
    return _process_images_in_threads(
        images,
        start_page=start_page,
        page_numbers=page_numbers,
        parallel=False,
    )


def ocr_single_page_image(
    image,
    page_number: int,
    include_rendered_image: bool = True,
) -> Dict[str, Any]:
    """
    Run OCR on a single PIL image.

    Fast path:
      - run PaddleOCR on the raw rendered page first
      - accept it immediately if it looks good enough
    Slow path:
      - run the full preprocessing pipeline only for pages that need it
    """
    warnings: List[str] = []
    raw_image = _page_array_from_image(image)
    rendered_image = raw_image if include_rendered_image else None
    fast_page: Optional[Dict[str, Any]] = None

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
            )
            if _ocr_result_looks_good(fast_page):
                return fast_page
    except Exception as exc:
        logger.debug("Fast OCR path failed on page %s: %s", page_number, exc)

    try:
        processed_arr, preprocess_meta = preprocess_page_image(image)
    except Exception as exc:
        logger.error(
            "Preprocessing failed on page %s: %s", page_number, exc, exc_info=True
        )
        warnings.append(f"Preprocessing failed: {exc}")
        processed_arr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        preprocess_meta = {}

    if preprocess_meta.get("deskew_angle", 0) != 0:
        warnings.append(f"Page was deskewed by {preprocess_meta['deskew_angle']:.1f}°")

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
    )

    # If the light pass also looks good, keep the better of the two results.
    if fast_page is not None and _ocr_result_looks_good(fast_page):
        fast_score = len(fast_page.get("text", "")) + float(
            fast_page.get("confidence", 0.0)
        ) * 100.0
        heavy_score = len(heavy_page.get("text", "")) + float(
            heavy_page.get("confidence", 0.0)
        ) * 100.0
        if fast_score >= heavy_score:
            return fast_page

    return heavy_page


def _process_images_in_threads(
    images: List,
    start_page: int,
    page_numbers: Optional[List[int]],
    parallel: bool = True,
) -> List[Dict[str, Any]]:
    """Shared page-processing helper for both path- and bytes-based inputs."""
    results: List[Dict[str, Any]] = []
    max_workers = (
        min(settings.effective_ocr_page_workers, len(images)) if len(images) > 1 else 1
    )

    def _process(item):
        idx, pil_image = item
        page_num = page_numbers[idx] if page_numbers else start_page + idx
        return ocr_single_page_image(pil_image, page_num)

    if not parallel or max_workers == 1:
        for item in enumerate(images):
            results.append(_process(item))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process, item) for item in enumerate(images)]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    results.sort(key=lambda item: item["page_number"])

    gc.collect()
    return results


def _extract_scanned_tables_from_page(
    rendered_image: np.ndarray,
    raw_results: List,
    page_number: int,
) -> List[Dict[str, Any]]:
    """Lightweight scanned table extraction that stays inside the worker process."""
    if not raw_results:
        return []

    try:
        from app.services.table_extractor import extract_tables_scanned
    except Exception as exc:
        logger.warning(
            "Scanned table extractor unavailable for page %s: %s",
            page_number,
            exc,
        )
        return []

    try:
        return extract_tables_scanned(rendered_image, raw_results, page_number)
    except Exception as exc:
        logger.warning(
            "Scanned table extraction failed page %s: %s", page_number, exc
        )
        return []


def extract_ocr_pdf(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    dpi: int = None,
) -> List[Dict[str, Any]]:
    """Full OCR extraction pipeline using the PDF on disk."""
    dpi = dpi or settings.OCR_DPI
    try:
        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
    except Exception:
        total_pages = 0

    normalized_pages = _normalize_page_numbers(page_numbers, total_pages) if total_pages else (page_numbers or [])
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
            "OCR using process pool | pages=%s | chunks=%s | workers=%s | dpi=%s",
            len(normalized_pages),
            len(chunks),
            settings.effective_ocr_chunk_workers,
            dpi,
        )
        try:
            ctx = mp.get_context("spawn")
            stat = pdf_path.stat()
            path_key = str(pdf_path)
            with ProcessPoolExecutor(
                max_workers=min(settings.effective_ocr_chunk_workers, len(chunks)),
                mp_context=ctx,
            ) as executor:
                futures = [
                    executor.submit(
                        _ocr_chunk_worker_from_path,
                        (path_key, stat.st_mtime_ns, stat.st_size, chunk, dpi),
                    )
                    for chunk in chunks
                ]
                results: List[Dict[str, Any]] = []
                for future in as_completed(futures):
                    results.extend(future.result())
        except Exception as exc:
            logger.warning(
                "Process pool OCR failed, falling back to sequential/threaded mode: %s",
                exc,
            )
            all_images = _render_pages_for_ocr(
                pdf_path=pdf_path,
                page_numbers=normalized_pages,
                dpi=dpi,
            )
            results = _process_images_in_threads(
                all_images,
                start_page=normalized_pages[0] if normalized_pages else 1,
                page_numbers=normalized_pages,
                parallel=True,
            )
    else:
        all_images = _render_pages_for_ocr(
            pdf_path=pdf_path,
            page_numbers=normalized_pages,
            dpi=dpi,
        )
        results = _process_images_in_threads(
            all_images,
            start_page=normalized_pages[0] if normalized_pages else 1,
            page_numbers=normalized_pages,
            parallel=True,
        )

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
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()
    except Exception:
        total_pages = 0

    normalized_pages = _normalize_page_numbers(page_numbers, total_pages) if total_pages else (page_numbers or [])
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
            "OCR using process pool | pages=%s | chunks=%s | workers=%s | dpi=%s",
            len(normalized_pages),
            len(chunks),
            settings.effective_ocr_chunk_workers,
            dpi,
        )
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(settings.effective_ocr_chunk_workers, len(chunks)),
                mp_context=ctx,
            ) as executor:
                futures = [
                    executor.submit(_ocr_chunk_worker, (pdf_bytes, chunk, dpi))
                    for chunk in chunks
                ]
                results: List[Dict[str, Any]] = []
                for future in as_completed(futures):
                    results.extend(future.result())
        except Exception as exc:
            logger.warning(
                "Process pool OCR failed, falling back to sequential/threaded mode: %s",
                exc,
            )
            all_images = _render_pages_for_ocr(
                pdf_bytes=pdf_bytes,
                page_numbers=normalized_pages,
                dpi=dpi,
            )
            results = _process_images_in_threads(
                all_images,
                start_page=normalized_pages[0] if normalized_pages else 1,
                page_numbers=normalized_pages,
                parallel=False,
            )
    else:
        all_images = _render_pages_for_ocr(
            pdf_bytes=pdf_bytes,
            page_numbers=normalized_pages,
            dpi=dpi,
        )
        results = _process_images_in_threads(
            all_images,
            start_page=normalized_pages[0] if normalized_pages else 1,
            page_numbers=normalized_pages,
            parallel=False,
        )

    logger.info("OCR extraction complete | pages=%s | bytes input", len(results))
    return results


def extract_ocr_pdf_local(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    dpi: int = None,
) -> List[Dict[str, Any]]:
    """
    Local OCR path intended for process-pool chunk workers.

    This performs rendering + OCR sequentially inside the worker process to
    avoid nested process pools and excess IPC overhead.
    """
    dpi = dpi or settings.OCR_DPI
    all_images = _render_pages_for_ocr(
        pdf_path=pdf_path,
        page_numbers=page_numbers,
        dpi=dpi,
    )
    start_page = min(page_numbers) if page_numbers else 1
    results: List[Dict[str, Any]] = []

    for idx, pil_image in enumerate(all_images):
        page_num = page_numbers[idx] if page_numbers else start_page + idx
        page_info = ocr_single_page_image(
            pil_image,
            page_num,
            include_rendered_image=False,
        )

        if page_info.get("text") or page_info.get("raw_results"):
            rendered_image = _page_array_from_image(pil_image)
            page_info["tables"] = _extract_scanned_tables_from_page(
                rendered_image,
                page_info.get("raw_results", []),
                page_num,
            )
        else:
            page_info["tables"] = []

        page_info.pop("rendered_image", None)
        results.append(page_info)

    results.sort(key=lambda item: item["page_number"])
    return results
