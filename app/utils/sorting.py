"""
Sorting Utilities
Implements correct reading-order sorting for text blocks extracted
from both digital PDFs (coordinates in pts) and OCR results (pixels).

Reading order: top-to-bottom, left-to-right (standard Western documents).
Uses a Y-tolerance band to group blocks on the same visual line.
"""

from typing import Any, Dict, List, Tuple

from app.config.constants import LINE_Y_TOLERANCE, LINE_Y_TOLERANCE_OCR


def sort_digital_blocks(
    blocks: List[Dict[str, Any]],
    y_tolerance: int = LINE_Y_TOLERANCE,
) -> List[Dict[str, Any]]:
    """
    Sort PyMuPDF text blocks in reading order.

    Each block dict must have keys: x0, y0, x1, y1, text.
    Strategy:
      1. Sort by y0 (top edge)
      2. Group blocks within y_tolerance of each other into a "row"
      3. Within each row, sort by x0 (left edge)
    """
    if not blocks:
        return []

    # Initial sort by top-edge
    sorted_by_y = sorted(blocks, key=lambda b: b["y0"])

    rows: List[List[Dict]] = []
    current_row: List[Dict] = [sorted_by_y[0]]
    current_y = sorted_by_y[0]["y0"]

    for block in sorted_by_y[1:]:
        if abs(block["y0"] - current_y) <= y_tolerance:
            current_row.append(block)
        else:
            rows.append(sorted(current_row, key=lambda b: b["x0"]))
            current_row = [block]
            current_y = block["y0"]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b["x0"]))

    return [block for row in rows for block in row]


def sort_ocr_results(
    ocr_results: List[Tuple],
    y_tolerance: int = LINE_Y_TOLERANCE_OCR,
) -> List[Tuple]:
    """
    Sort PaddleOCR results in reading order.

    PaddleOCR returns: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, conf)), ...]
    We extract the top-left y coordinate for sorting.

    Returns sorted list in the same PaddleOCR format.
    """
    if not ocr_results:
        return []

    def top_y(result):
        box = result[0]  # 4-point bounding box
        return min(pt[1] for pt in box)

    def left_x(result):
        box = result[0]
        return min(pt[0] for pt in box)

    sorted_by_y = sorted(ocr_results, key=top_y)

    rows: List[List] = []
    current_row = [sorted_by_y[0]]
    current_y = top_y(sorted_by_y[0])

    for item in sorted_by_y[1:]:
        if abs(top_y(item) - current_y) <= y_tolerance:
            current_row.append(item)
        else:
            rows.append(sorted(current_row, key=left_x))
            current_row = [item]
            current_y = top_y(item)

    if current_row:
        rows.append(sorted(current_row, key=left_x))

    return [item for row in rows for item in row]


def group_into_paragraphs(
    text_lines: List[str],
    gap_threshold: int = 2,
) -> List[str]:
    """
    Merge consecutive short lines that appear to form a single paragraph.
    Lines separated by empty lines become separate paragraphs.

    Args:
        text_lines: List of extracted text lines.
        gap_threshold: Consecutive blank lines that signal a new paragraph.

    Returns:
        List of paragraph strings.
    """
    paragraphs: List[str] = []
    buffer: List[str] = []
    blank_count = 0

    for line in text_lines:
        stripped = line.strip()
        if stripped:
            if blank_count >= gap_threshold and buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            buffer.append(stripped)
            blank_count = 0
        else:
            blank_count += 1

    if buffer:
        paragraphs.append(" ".join(buffer))

    return paragraphs


def merge_hyphenated_lines(lines: List[str]) -> List[str]:
    """
    Merge lines ending with a hyphen (word broken across lines).
    e.g. ["adminis-", "tration"] → ["administration"]
    """
    merged: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith("-") and i + 1 < len(lines):
            merged.append(line[:-1] + lines[i + 1].lstrip())
            i += 2
        else:
            merged.append(line)
            i += 1
    return merged
