"""
Table Extraction Engine

Fixes applied:
  M8  — extract_tables_digital() always used bytes path when bytes available;
         _pdfplumber_extract() (disk path) is now only called as true fallback.
  M11 — accuracy field is now computed from row/column counts instead of
         always being None.
  I3  — extract_tables_scanned() now detects and returns ALL table grids on a
         page (previously always returned exactly one table with index 0).
  FIX — Added extract_tables_digital_batch() — opens pdfplumber ONCE for all
         requested pages, drastically reducing open() call overhead (was O(n) opens).
  FIX — _pdfplumber_extract_from_bytes() now accepts optional page_num=None
         to return all pages when called without a specific page number.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config.constants import (
    MIN_TABLE_COLS,
    MIN_TABLE_ROWS,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_table_accuracy(headers: List[str], rows: List[List[str]]) -> float:
    """
    M11 fix: estimate table accuracy from fill rate.
    Returns fraction of non-empty cells across headers + rows.
    """
    all_cells = list(headers) + [cell for row in rows for cell in row]
    if not all_cells:
        return 0.0
    filled = sum(1 for c in all_cells if c and c.strip())
    return round(filled / len(all_cells), 4)


def _pdfplumber_extract(pdf_path: Path, page_num: int) -> List[Dict]:
    """Digital PDF extraction from disk path (fallback when bytes unavailable)."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed.")
        return []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num > len(pdf.pages):
                return []
            page = pdf.pages[page_num - 1]
            raw_tables = page.extract_tables() or []
    except Exception as exc:
        logger.error("pdfplumber failed on page %s: %s", page_num, exc)
        return []

    extracted: List[Dict] = []
    for i, tbl in enumerate(raw_tables):
        if not tbl or len(tbl) < MIN_TABLE_ROWS:
            continue
        if not tbl[0] or len(tbl[0]) < MIN_TABLE_COLS:
            continue

        headers = [str(c) if c else "" for c in tbl[0]]
        rows = [[str(c) if c else "" for c in row] for row in tbl[1:]]

        extracted.append(
            {
                "page": page_num,
                "table_index": i,
                "headers": headers,
                "rows": rows,
                "extraction_method": "pdfplumber",
                "accuracy": _compute_table_accuracy(headers, rows),
            }
        )

    return extracted


def _pdfplumber_extract_from_bytes(
    pdf_bytes: bytes,
    page_num: Optional[int] = None,
) -> List[Dict]:
    """
    FIX: Byte-based pdfplumber extraction — preferred over disk path.
    page_num is now Optional. When None, extracts from all pages (used in batch mode).
    When provided, extracts from that specific 1-indexed page only.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed.")
        return []

    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            if page_num is not None:
                if page_num > len(pdf.pages):
                    return []
                pages_to_process = [(page_num, pdf.pages[page_num - 1])]
            else:
                pages_to_process = [(i + 1, page) for i, page in enumerate(pdf.pages)]

            extracted: List[Dict] = []
            for pnum, page in pages_to_process:
                raw_tables = page.extract_tables() or []
                for i, tbl in enumerate(raw_tables):
                    if not tbl or len(tbl) < MIN_TABLE_ROWS:
                        continue
                    if not tbl[0] or len(tbl[0]) < MIN_TABLE_COLS:
                        continue

                    headers = [str(c) if c else "" for c in tbl[0]]
                    rows = [[str(c) if c else "" for c in row] for row in tbl[1:]]

                    extracted.append(
                        {
                            "page": pnum,
                            "table_index": i,
                            "headers": headers,
                            "rows": rows,
                            "extraction_method": "pdfplumber",
                            "accuracy": _compute_table_accuracy(headers, rows),
                        }
                    )
            return extracted

    except Exception as exc:
        logger.error("pdfplumber(bytes) failed: %s", exc)
        return []


def extract_tables_digital(
    pdf_path: Path,
    page_num: int,
    pdf_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    """
    Extract tables from a digital PDF page.
    M8 fix: always prefer the bytes path when bytes are available.
    """
    if pdf_bytes is not None:
        return _pdfplumber_extract_from_bytes(pdf_bytes, page_num)
    return _pdfplumber_extract(pdf_path, page_num)


def extract_tables_digital_batch(
    pdf_path: Path,
    page_numbers: List[int],
    pdf_bytes: Optional[bytes] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    FIX: Extract tables from multiple pages by opening pdfplumber ONCE.

    Previously, the pipeline called extract_tables_digital() per page which
    opened a new pdfplumber handle for every page — O(n) expensive file opens.
    This function opens the PDF once and extracts all requested pages in a
    single pass, reducing overhead from O(n) to O(1) file opens.

    Returns a dict mapping page_number → list of table dicts.
    """
    result: Dict[int, List[Dict[str, Any]]] = {p: [] for p in page_numbers}

    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed — skipping batch table extraction.")
        return result

    page_set = set(page_numbers)

    try:
        source = BytesIO(pdf_bytes) if pdf_bytes is not None else str(pdf_path)
        with pdfplumber.open(source) as pdf:
            for page_num in sorted(page_set):
                if page_num < 1 or page_num > len(pdf.pages):
                    continue
                try:
                    raw_tables = pdf.pages[page_num - 1].extract_tables() or []
                except Exception as exc:
                    logger.warning("pdfplumber batch failed page %s: %s", page_num, exc)
                    continue

                for i, tbl in enumerate(raw_tables):
                    if not tbl or len(tbl) < MIN_TABLE_ROWS:
                        continue
                    if not tbl[0] or len(tbl[0]) < MIN_TABLE_COLS:
                        continue
                    headers = [str(c) if c else "" for c in tbl[0]]
                    rows = [[str(c) if c else "" for c in row] for row in tbl[1:]]
                    result[page_num].append(
                        {
                            "page": page_num,
                            "table_index": i,
                            "headers": headers,
                            "rows": rows,
                            "extraction_method": "pdfplumber",
                            "accuracy": _compute_table_accuracy(headers, rows),
                        }
                    )
    except Exception as exc:
        logger.error("pdfplumber batch open failed: %s", exc)

    return result


def _detect_table_grid(
    binary_image: np.ndarray,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Detect horizontal and vertical lines in a binary image.
    Returns (row_y_positions, col_x_positions) or None if no grid found.
    """
    h, w = binary_image.shape[:2]

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 4, 1), 1))
    h_morph = cv2.morphologyEx(cv2.bitwise_not(binary_image), cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 4, 1)))
    v_morph = cv2.morphologyEx(cv2.bitwise_not(binary_image), cv2.MORPH_OPEN, v_kernel)

    h_projection = np.sum(h_morph > 128, axis=1)
    row_ys = [y for y in range(h) if h_projection[y] > w * 0.5]

    v_projection = np.sum(v_morph > 128, axis=0)
    col_xs = [x for x in range(w) if v_projection[x] > h * 0.5]

    def cluster(positions, gap=10):
        clusters = []
        for p in sorted(set(positions)):
            if not clusters or p - clusters[-1] > gap:
                clusters.append(p)
        return clusters

    row_ys = cluster(row_ys)
    col_xs = cluster(col_xs)

    if len(row_ys) < MIN_TABLE_ROWS or len(col_xs) < MIN_TABLE_COLS:
        return None

    return row_ys, col_xs


def _map_ocr_to_cells(
    ocr_results: List,
    row_ys: List[int],
    col_xs: List[int],
) -> List[List[str]]:
    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1
    grid: List[List[List[str]]] = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

    for item in ocr_results:
        if not item or len(item) < 2:
            continue
        box, (text, _) = item
        cx = sum(pt[0] for pt in box) / 4
        cy = sum(pt[1] for pt in box) / 4

        row_idx = None
        for r in range(n_rows):
            if row_ys[r] <= cy < row_ys[r + 1]:
                row_idx = r
                break

        col_idx = None
        for c in range(n_cols):
            if col_xs[c] <= cx < col_xs[c + 1]:
                col_idx = c
                break

        if row_idx is not None and col_idx is not None:
            grid[row_idx][col_idx].append(text)

    return [[" ".join(cell) for cell in row] for row in grid]


def _split_grids_by_gap(row_ys: List[int], gap_threshold: int = 30) -> List[List[int]]:
    """
    I3 fix: split a flat list of row positions into separate table regions
    wherever there is a gap larger than gap_threshold pixels.
    """
    if not row_ys:
        return []
    groups: List[List[int]] = [[row_ys[0]]]
    for y in row_ys[1:]:
        if y - groups[-1][-1] > gap_threshold:
            groups.append([y])
        else:
            groups[-1].append(y)
    return groups


def extract_tables_scanned(
    preprocessed_image: np.ndarray,
    ocr_results: List,
    page_num: int,
) -> List[Dict[str, Any]]:
    """
    Extract tables from a scanned page using grid detection + OCR tokens.

    I3 fix: detects ALL tables on the page (previously returned only one).
    M11 fix: accuracy field is now computed from cell fill rate.
    """
    if len(preprocessed_image.shape) == 3:
        gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = preprocessed_image

    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    grid = _detect_table_grid(binary)
    if grid is None:
        logger.debug("Page %s: No table grid detected in scanned page.", page_num)
        return []

    all_row_ys, col_xs = grid

    # I3 fix: split row positions into separate table regions by vertical gap
    row_groups = _split_grids_by_gap(all_row_ys)

    extracted: List[Dict[str, Any]] = []
    for table_index, row_ys in enumerate(row_groups):
        if len(row_ys) < MIN_TABLE_ROWS:
            continue

        cell_data = _map_ocr_to_cells(ocr_results, row_ys, col_xs)
        if not cell_data or len(cell_data) < MIN_TABLE_ROWS:
            continue

        headers = cell_data[0]
        rows = cell_data[1:]

        extracted.append(
            {
                "page": page_num,
                "table_index": table_index,
                "headers": headers,
                "rows": rows,
                "extraction_method": "ocr-grid",
                "accuracy": _compute_table_accuracy(headers, rows),
            }
        )

    if extracted:
        logger.info(
            "Page %s: Extracted %s scanned table(s)",
            page_num,
            len(extracted),
        )
    return extracted
