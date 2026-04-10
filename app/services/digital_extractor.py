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

import re
from io import BytesIO
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
    Extract all text blocks from a page with full coordinate data.
    Uses 'dict' mode to preserve block/line/span hierarchy.
    """
    try:
        raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    except Exception:
        return []
    blocks = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # skip image blocks
            continue

        block_text_parts: List[str] = []
        font_sizes: List[float] = []
        is_bold = False

        for line in block.get("lines", []):
            line_parts: List[str] = []
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    line_parts.append(text)
                    font_sizes.append(span.get("size", 12.0))
                    flags = span.get("flags", 0)
                    if flags & 2**4:  # bold flag in PyMuPDF
                        is_bold = True
            if line_parts:
                block_text_parts.append(" ".join(line_parts))

        full_text = "\n".join(block_text_parts).strip()

        # Skip blocks that are too short (likely noise/artifacts)
        if len(full_text) < MIN_BLOCK_CHAR_COUNT:
            continue

        bbox = block["bbox"]
        blocks.append(
            {
                "x0": bbox[0],
                "y0": bbox[1],
                "x1": bbox[2],
                "y1": bbox[3],
                "text": full_text,
                "avg_font_size": sum(font_sizes) / len(font_sizes)
                if font_sizes
                else 12.0,
                "is_bold": is_bold,
            }
        )

    return blocks


def _fallback_page_text(pdf_bytes: bytes) -> str:
    """
    Final fallback for malformed PDFs. Extract visible strings from raw content
    streams or any literal text tokens in the file.
    """
    try:
        return extract_text_from_pdf_bytes(pdf_bytes)
    except Exception:
        return ""


def extract_digital_page(
    page: fitz.Page,
    page_number: int,
    fallback_text: str = "",
) -> Dict[str, Any]:
    """
    Extract and sort text from a single digital PDF page.

    Returns dict with:
      text: clean extracted text
      blocks: sorted raw blocks (for downstream use)
      warnings: list of warning strings
    """
    warnings: List[str] = []

    blocks = _extract_page_blocks(page)

    if not blocks:
        if fallback_text.strip():
            warnings.append(
                f"Page {page_number}: PyMuPDF fallback extraction used."
            )
            synthetic = _build_synthetic_block(fallback_text)
            return {
                "text": fallback_text.strip(),
                "blocks": [synthetic],
                "warnings": warnings,
            }

        warnings.append(f"Page {page_number}: no extractable text blocks found.")
        return {"text": "", "blocks": [], "warnings": warnings}

    # Sort into correct reading order
    sorted_blocks = sort_digital_blocks(blocks)

    # Extract lines, merge hyphenated words
    all_lines: List[str] = []
    for block in sorted_blocks:
        lines = block["text"].split("\n")
        all_lines.extend(lines)

    all_lines = merge_hyphenated_lines(all_lines)

    # Join with newlines; double-newline between blocks for paragraph spacing
    paragraph_texts = []
    current_para: List[str] = []
    prev_y1: Optional[float] = None

    for block in sorted_blocks:
        if (
            prev_y1 is not None
            and (block["y0"] - prev_y1) > block["avg_font_size"] * 1.5
        ):
            # Large gap → new paragraph
            if current_para:
                paragraph_texts.append(" ".join(current_para))
                current_para = []
        current_para.extend(block["text"].split("\n"))
        prev_y1 = block["y1"]

    if current_para:
        paragraph_texts.append(" ".join(current_para))

    final_text = "\n\n".join(paragraph_texts).strip()

    if not final_text.strip() and fallback_text.strip():
        final_text = fallback_text.strip()
        sorted_blocks = [_build_synthetic_block(final_text)]

    return {
        "text": final_text,
        "blocks": sorted_blocks,
        "warnings": warnings,
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

    fallback_text = ""
    source_bytes = pdf_bytes
    if source_bytes is None:
        try:
            source_bytes = pdf_path.read_bytes()
        except Exception:
            source_bytes = None

    if source_bytes is not None:
        try:
            fallback_text = _fallback_page_text(source_bytes)
        except Exception:
            fallback_text = ""

    for page_num in targets:
        if page_num < 1 or page_num > total:
            logger.warning(f"Page {page_num} out of range (1..{total}), skipping.")
            continue

        page = doc[page_num - 1]
        try:
            result = extract_digital_page(page, page_num, fallback_text=fallback_text)
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
