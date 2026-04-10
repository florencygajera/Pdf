"""
Digital Extraction Engine
Extracts text from digital (text-based) PDF pages using PyMuPDF.

Key capabilities:
- Block-level extraction with precise coordinates
- Reading-order sorting (y → x)
- Hyphenation merging
- Empty-page detection
- Font metadata preservation (for heading detection)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from app.config.constants import MIN_BLOCK_CHAR_COUNT
from app.utils.logger import get_logger
from app.utils.pdf_text_fallback import extract_text_from_pdf_bytes
from app.utils.sorting import (
    merge_hyphenated_lines,
    sort_digital_blocks,
)

logger = get_logger(__name__)

try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass


def _build_synthetic_block(text: str) -> Dict[str, Any]:
    return {
        "x0": 0.0,
        "y0": 0.0,
        "x1": 0.0,
        "y1": 0.0,
        "text": text.strip(),
        "avg_font_size": 12.0,
        "is_bold": False,
    }


def _extract_page_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Extract text blocks from a page with coordinate data.

    `page.get_text("blocks", sort=True)` is lighter than parsing the full
    span dictionary and still preserves enough geometry for reading-order
    reconstruction.
    """
    try:
        raw_blocks = page.get_text("blocks", sort=True)
    except Exception:
        return []

    blocks: List[Dict[str, Any]] = []

    for block in raw_blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        full_text = str(text).strip()
        if len(full_text) < MIN_BLOCK_CHAR_COUNT:
            continue
        blocks.append(
            {
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "text": full_text,
                "avg_font_size": 12.0,
                "is_bold": False,
            }
        )

    return blocks


def extract_digital_page(
    page: fitz.Page,
    page_number: int,
    pdf_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Extract and sort text from a single digital PDF page.

    Returns dict with:
      text: clean extracted text
      blocks: sorted raw blocks (for downstream use)
      warnings: list of warning strings
    """
    warnings: List[str] = []

    # Fast path: PyMuPDF's plain text extraction is usually the cleanest and
    # cheapest representation when the PDF already contains searchable text.
    try:
        raw_text = page.get_text("text").strip()
    except Exception:
        raw_text = ""

    blocks = _extract_page_blocks(page)
    sorted_blocks = sort_digital_blocks(blocks) if blocks else []

    if sorted_blocks:
        paragraph_texts: List[str] = []
        current_para: List[str] = []
        prev_y1: Optional[float] = None

        for block in sorted_blocks:
            block_lines = [line.strip() for line in block["text"].splitlines() if line.strip()]
            if not block_lines:
                continue
            gap = block["y0"] - prev_y1 if prev_y1 is not None else 0.0
            if prev_y1 is not None and gap > max(12.0, block["avg_font_size"] * 1.3):
                if current_para:
                    paragraph_texts.append(" ".join(current_para))
                    current_para = []
            current_para.extend(block_lines)
            prev_y1 = block["y1"]

        if current_para:
            paragraph_texts.append(" ".join(current_para))

        final_text = "\n\n".join(paragraph_texts).strip()
        final_text = merge_hyphenated_lines(final_text.splitlines())
        final_text = "\n".join(final_text).strip()
    else:
        final_text = raw_text

    if not final_text and raw_text:
        final_text = raw_text
        if not sorted_blocks:
            sorted_blocks = [_build_synthetic_block(final_text)]

    if not final_text.strip() and pdf_bytes is not None:
        try:
            fallback_text = extract_text_from_pdf_bytes(pdf_bytes)
        except Exception:
            fallback_text = ""
        if fallback_text.strip():
            warnings.append(f"Page {page_number}: PyMuPDF fallback extraction used.")
            final_text = fallback_text.strip()
            sorted_blocks = [_build_synthetic_block(final_text)]

    if not final_text.strip():
        warnings.append(f"Page {page_number}: no extractable text blocks found.")
        return {
            "text": "",
            "blocks": [],
            "warnings": warnings,
            "page_width": float(page.rect.width),
        }

    return {
        "text": final_text,
        "blocks": sorted_blocks,
        "warnings": warnings,
        "page_width": float(page.rect.width),
    }


def extract_digital_pdf(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    pdf_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    """
    Extract text from all (or specified) pages of a digital PDF.

    Args:
        pdf_path: Path to PDF file.
        page_numbers: 1-indexed list of pages to extract. None = all.

    Returns:
        List of per-page result dicts (keys: page_number, text, blocks, warnings).

    Raises:
        ValueError: If PDF is corrupt or unreadable.
    """
    try:
        if pdf_bytes is not None:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        else:
            doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if doc.is_encrypted:
        if not doc.authenticate(""):
            raise ValueError("PDF is password protected.")

    total = len(doc)
    targets = page_numbers or list(range(1, total + 1))
    results: List[Dict[str, Any]] = []

    if pdf_bytes is None:
        try:
            pdf_bytes = pdf_path.read_bytes()
        except Exception:
            pdf_bytes = None

    for page_num in targets:
        if page_num < 1 or page_num > total:
            logger.warning(f"Page {page_num} out of range (1..{total}), skipping.")
            continue

        page = doc[page_num - 1]
        try:
            result = extract_digital_page(page, page_num, pdf_bytes=pdf_bytes)
        except Exception as exc:
            logger.error(f"Error extracting page {page_num}: {exc}", exc_info=True)
            result = {
                "text": "",
                "blocks": [],
                "warnings": [f"Extraction failed: {exc}"],
            }

        result["page_number"] = page_num
        results.append(result)
        logger.debug(f"Extracted page {page_num}/{total} | chars={len(result['text'])}")

    doc.close()
    logger.info(
        f"Digital extraction complete | pages={len(results)} | file={pdf_path.name}"
    )
    return results
