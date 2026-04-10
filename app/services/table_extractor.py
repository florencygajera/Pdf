"""
Table Extraction Engine

The production path keeps table extraction safe and warning-free by relying on
pdfplumber for digital PDFs and OpenCV/OCR grid detection for scanned pages.
Camelot is intentionally avoided here because its legacy PDF stack can emit
PyPDF2 deprecation warnings in modern environments.
"""

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


def _pdfplumber_extract(pdf_path: Path, page_num: int) -> List[Dict]:
    """
    Digital PDF fallback using pdfplumber only.

    This keeps the extraction path stable and avoids the Camelot/PyPDF2
    dependency chain that was emitting deprecation warnings.
    """
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
                "accuracy": None,
            }
        )

    return extracted


def extract_tables_digital(
    pdf_path: Path,
    page_num: int,
) -> List[Dict[str, Any]]:
    """Extract tables from a digital PDF page."""
    return _pdfplumber_extract(pdf_path, page_num)


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


def extract_tables_scanned(
    preprocessed_image: np.ndarray,
    ocr_results: List,
    page_num: int,
) -> List[Dict[str, Any]]:
    """Extract tables from a scanned page using grid detection + OCR tokens."""
    if len(preprocessed_image.shape) == 3:
        gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = preprocessed_image

    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    grid = _detect_table_grid(binary)
    if grid is None:
        logger.debug("Page %s: No table grid detected in scanned page.", page_num)
        return []

    row_ys, col_xs = grid
    cell_data = _map_ocr_to_cells(ocr_results, row_ys, col_xs)

    if not cell_data or len(cell_data) < MIN_TABLE_ROWS:
        return []

    headers = cell_data[0]
    rows = cell_data[1:]

    table = {
        "page": page_num,
        "table_index": 0,
        "headers": headers,
        "rows": rows,
        "extraction_method": "ocr-grid",
        "accuracy": None,
    }

    logger.info(
        "Page %s: Extracted scanned table | rows=%s | cols=%s",
        page_num,
        len(rows),
        len(headers),
    )
    return [table]

