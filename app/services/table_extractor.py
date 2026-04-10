"""
Table Extraction Engine
Handles two scenarios:
  1. Digital PDFs → Camelot (lattice mode first, flavor fallback) + pdfplumber
  2. Scanned PDFs → OpenCV grid detection → OCR tokens mapped to cells

Outputs structured TableData-compatible dicts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config.constants import (
    MIN_TABLE_COLS,
    MIN_TABLE_ROWS,
    TABLE_SCORE_THRESHOLD,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DIGITAL TABLE EXTRACTION (Camelot + pdfplumber fallback)
# ═══════════════════════════════════════════════════════════════════════════════


def _camelot_extract(
    pdf_path: Path, page_num: int, flavor: str = "lattice"
) -> List[Dict]:
    """
    Extract tables from a digital PDF page using Camelot.
    Flavor: 'lattice' (ruled tables) or 'stream' (whitespace-based).
    """
    try:
        import camelot

        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_num),
            flavor=flavor,
        )
    except ImportError:
        logger.warning("Camelot not installed. Falling back to pdfplumber.")
        return []
    except Exception as exc:
        logger.warning(f"Camelot ({flavor}) failed on page {page_num}: {exc}")
        return []

    extracted = []
    for i, table in enumerate(tables):
        if table.accuracy < TABLE_SCORE_THRESHOLD * 100:
            logger.debug(
                f"Page {page_num} table {i}: low accuracy {table.accuracy:.1f}%, skipping."
            )
            continue

        df = table.df
        if df.shape[0] < MIN_TABLE_ROWS or df.shape[1] < MIN_TABLE_COLS:
            continue

        rows = df.values.tolist()
        headers = [str(h) for h in rows[0]] if rows else []
        data_rows = [[str(c) for c in row] for row in rows[1:]]

        extracted.append(
            {
                "page": page_num,
                "table_index": i,
                "headers": headers,
                "rows": data_rows,
                "extraction_method": f"camelot-{flavor}",
                "accuracy": table.accuracy,
            }
        )

    return extracted


def _pdfplumber_extract(pdf_path: Path, page_num: int) -> List[Dict]:
    """
    Fallback: pdfplumber table extraction.
    Works better for borderless/stream tables.
    """
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num > len(pdf.pages):
                return []
            page = pdf.pages[page_num - 1]
            raw_tables = page.extract_tables()

        extracted = []
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

    except ImportError:
        logger.warning("pdfplumber not installed.")
        return []
    except Exception as exc:
        logger.error(f"pdfplumber failed on page {page_num}: {exc}")
        return []


def extract_tables_digital(
    pdf_path: Path,
    page_num: int,
) -> List[Dict[str, Any]]:
    """
    Extract tables from a digital PDF page.
    Tries Camelot lattice → Camelot stream → pdfplumber.
    """
    # Try lattice (ruled borders first)
    tables = _camelot_extract(pdf_path, page_num, flavor="lattice")
    if tables:
        return tables

    # Try stream (whitespace detection)
    tables = _camelot_extract(pdf_path, page_num, flavor="stream")
    if tables:
        return tables

    # Fallback
    return _pdfplumber_extract(pdf_path, page_num)


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNED TABLE EXTRACTION (OpenCV line detection + OCR cell mapping)
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_table_grid(
    binary_image: np.ndarray,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Detect horizontal and vertical lines in a binary image.
    Returns (row_y_positions, col_x_positions) or None if no grid found.
    """
    h, w = binary_image.shape[:2]

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    h_morph = cv2.morphologyEx(cv2.bitwise_not(binary_image), cv2.MORPH_OPEN, h_kernel)

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    v_morph = cv2.morphologyEx(cv2.bitwise_not(binary_image), cv2.MORPH_OPEN, v_kernel)

    # Find horizontal line y-positions
    h_projection = np.sum(h_morph > 128, axis=1)
    row_ys = [y for y in range(h) if h_projection[y] > w * 0.5]

    # Find vertical line x-positions
    v_projection = np.sum(v_morph > 128, axis=0)
    col_xs = [x for x in range(w) if v_projection[x] > h * 0.5]

    # Cluster nearby lines
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
    """
    Map OCR tokens to the nearest grid cell based on centroid position.
    Returns a 2D list: rows × cols.
    """
    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1
    grid: List[List[List[str]]] = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

    for item in ocr_results:
        if not item or len(item) < 2:
            continue
        box, (text, _) = item
        # Centroid of bounding box
        cx = sum(pt[0] for pt in box) / 4
        cy = sum(pt[1] for pt in box) / 4

        # Find which row
        row_idx = None
        for r in range(n_rows):
            if row_ys[r] <= cy < row_ys[r + 1]:
                row_idx = r
                break

        # Find which column
        col_idx = None
        for c in range(n_cols):
            if col_xs[c] <= cx < col_xs[c + 1]:
                col_idx = c
                break

        if row_idx is not None and col_idx is not None:
            grid[row_idx][col_idx].append(text)

    # Flatten cell token lists to strings
    return [[" ".join(cell) for cell in row] for row in grid]


def extract_tables_scanned(
    preprocessed_image: np.ndarray,
    ocr_results: List,
    page_num: int,
) -> List[Dict[str, Any]]:
    """
    Extract tables from a scanned page using OpenCV grid detection + OCR token mapping.

    Args:
        preprocessed_image: Grayscale/binary page image (numpy array).
        ocr_results: Filtered PaddleOCR results from this page.
        page_num: 1-indexed page number.

    Returns:
        List of table dicts.
    """
    # Convert to grayscale for processing
    if len(preprocessed_image.shape) == 3:
        gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = preprocessed_image

    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    grid = _detect_table_grid(binary)
    if grid is None:
        logger.debug(f"Page {page_num}: No table grid detected in scanned page.")
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
        f"Page {page_num}: Extracted scanned table | "
        f"rows={len(rows)} | cols={len(headers)}"
    )
    return [table]
