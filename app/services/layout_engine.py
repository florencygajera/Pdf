"""
Layout Reconstruction Engine
Reconstructs correct multi-column reading order for complex document layouts.

Uses column detection heuristics for:
  - Multi-column government documents
  - Mixed single+multi-column pages (common in gazettes/circulars)

Optionally uses LayoutParser for ML-based block classification (if installed).
"""

from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from app.config.constants import LINE_Y_TOLERANCE
from app.utils.logger import get_logger
from app.utils.sorting import sort_digital_blocks

logger = get_logger(__name__)
_layout_model = None
_layout_model_lock = Lock()

# Threshold: fraction of page width — gaps wider than this suggest column boundary
COLUMN_GAP_THRESHOLD = 0.08


def _detect_columns(
    blocks: List[Dict[str, Any]],
    page_width: float,
) -> int:
    """
    Estimate number of columns based on x-coordinate distribution of blocks.
    Returns 1 (single) or 2 (double column).
    """
    if not blocks or page_width == 0:
        return 1

    # X-center of each block
    x_centers = [(b["x0"] + b["x1"]) / 2 for b in blocks]
    if not x_centers:
        return 1

    mid = page_width / 2
    left_count = sum(1 for x in x_centers if x < mid)
    right_count = sum(1 for x in x_centers if x >= mid)

    # If both halves have meaningful content, it's 2-column
    if left_count > 0 and right_count > 0:
        ratio = min(left_count, right_count) / max(left_count, right_count)
        if ratio > 0.3:
            return 2

    return 1


def _split_into_columns(
    blocks: List[Dict[str, Any]],
    page_width: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split blocks into left and right columns.
    Uses page midpoint as separator.
    """
    mid = page_width / 2
    left = [b for b in blocks if (b["x0"] + b["x1"]) / 2 < mid]
    right = [b for b in blocks if (b["x0"] + b["x1"]) / 2 >= mid]
    return left, right


def reconstruct_reading_order(
    blocks: List[Dict[str, Any]],
    page_width: float = 612.0,  # A4/Letter default
    force_columns: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Reconstruct correct reading order for a page's text blocks.

    Algorithm:
      1. Detect layout (single vs multi-column)
      2. For multi-column: sort left column top-to-bottom, then right column
      3. For single-column: standard y→x sort

    Args:
        blocks: List of block dicts with x0, y0, x1, y1, text.
        page_width: Page width in points (used for column split).
        force_columns: Override column detection (1 or 2).

    Returns:
        Blocks in correct reading order.
    """
    if not blocks:
        return []

    n_cols = force_columns or _detect_columns(blocks, page_width)

    if n_cols == 1:
        return sort_digital_blocks(blocks)

    # 2-column layout
    left_blocks, right_blocks = _split_into_columns(blocks, page_width)
    sorted_left = sort_digital_blocks(left_blocks)
    sorted_right = sort_digital_blocks(right_blocks)

    logger.debug(
        f"2-column layout detected | left={len(left_blocks)} | right={len(right_blocks)}"
    )

    # Read left column fully, then right column
    return sorted_left + sorted_right


def try_layoutparser(
    image,
    page_number: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Optional: Use LayoutParser + Detectron2 for ML-based layout analysis.
    Returns list of region dicts {type, bbox, text} or None if unavailable.

    Types: 'Text', 'Title', 'List', 'Table', 'Figure'
    """
    try:
        import layoutparser as lp  # type: ignore
        import numpy as np
        from PIL import Image

        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        global _layout_model
        if _layout_model is None:
            with _layout_model_lock:
                if _layout_model is None:
                    _layout_model = lp.Detectron2LayoutModel(
                        "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                        label_map={
                            0: "Text",
                            1: "Title",
                            2: "List",
                            3: "Table",
                            4: "Figure",
                        },
                    )

        layout = _layout_model.detect(img_array)

        regions = []
        for block in layout:
            coords = block.coordinates  # (x1, y1, x2, y2)
            regions.append(
                {
                    "type": block.type,
                    "x0": coords[0],
                    "y0": coords[1],
                    "x1": coords[2],
                    "y1": coords[3],
                    "score": block.score,
                }
            )

        logger.info(
            f"LayoutParser detected {len(regions)} regions on page {page_number}"
        )
        return regions

    except ImportError:
        logger.debug("LayoutParser not installed — skipping ML layout detection.")
        return None
    except Exception as exc:
        logger.warning(f"LayoutParser failed on page {page_number}: {exc}")
        return None


def merge_text_blocks_to_paragraphs(
    blocks: List[Dict[str, Any]],
    avg_line_height: float = 14.0,
) -> str:
    """
    Convert sorted blocks into a clean string with paragraph breaks.
    Blocks separated by more than 1.5x line height become new paragraphs.
    """
    if not blocks:
        return ""

    paragraphs: List[str] = []
    current: List[str] = [blocks[0]["text"]]
    prev_y1 = blocks[0]["y1"]

    for block in blocks[1:]:
        gap = block["y0"] - prev_y1
        if gap > avg_line_height * 1.5:
            paragraphs.append(" ".join(current))
            current = [block["text"]]
        else:
            current.append(block["text"])
        prev_y1 = block["y1"]

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)
