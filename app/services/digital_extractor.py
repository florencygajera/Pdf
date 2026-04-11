"""
Digital Extraction Engine
Extracts text from digital (text-based) PDF pages using PyMuPDF.

Key capabilities:
- Block-level extraction with precise coordinates
- Reading-order sorting (y → x)
- Hyphenation merging
- Empty-page detection
- Font metadata preservation (for heading detection)

Fixes applied:
  M9 — doc.close() is now always called in a try/finally block.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import re
import fitz  # PyMuPDF

from app.config.constants import MIN_BLOCK_CHAR_COUNT
from app.utils.logger import get_logger
from app.utils.pdf_text_fallback import extract_text_from_pdf_bytes
from app.utils.sorting import (
    merge_hyphenated_lines,
    sort_digital_blocks,
)

logger = get_logger(__name__)

_STREAM_LITERAL_RE = re.compile(rb"\((?:\\.|[^\\)])*\)")
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

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


def _extract_words_text(page: fitz.Page) -> str:
    """Reconstruct text from word boxes when plain text extraction is weak."""
    try:
        words = page.get_text("words", sort=True)
    except Exception:
        return ""

    if not words:
        return ""

    items = []
    heights = []
    for word in words:
        if len(word) < 5:
            continue
        x0, y0, x1, y1, text = word[:5]
        token = str(text).strip()
        if not token:
            continue
        items.append((float(x0), float(y0), float(x1), float(y1), token))
        heights.append(max(1.0, float(y1) - float(y0)))

    if not items:
        return ""

    items.sort(key=lambda item: (item[1], item[0]))
    line_tolerance = max(3.0, sorted(heights)[len(heights) // 2] * 0.7)

    lines: List[Dict[str, Any]] = []
    current_words: List[tuple] = [items[0]]
    current_top = items[0][1]
    current_bottom = items[0][3]

    for item in items[1:]:
        x0, y0, x1, y1, _ = item
        if (
            y0 <= current_bottom + line_tolerance
            or abs(y0 - current_top) <= line_tolerance
        ):
            current_words.append(item)
            current_bottom = max(current_bottom, y1)
        else:
            lines.append(
                {
                    "words": sorted(current_words, key=lambda w: w[0]),
                    "top": current_top,
                    "bottom": current_bottom,
                }
            )
            current_words = [item]
            current_top = y0
            current_bottom = y1

    if current_words:
        lines.append(
            {
                "words": sorted(current_words, key=lambda w: w[0]),
                "top": current_top,
                "bottom": current_bottom,
            }
        )

    paragraphs: List[str] = []
    current_para: List[str] = []
    avg_height = sorted(heights)[len(heights) // 2] if heights else 12.0

    for idx, line in enumerate(lines):
        line_text = " ".join(word[4] for word in line["words"]).strip()
        if not line_text:
            continue

        if current_para:
            gap = line["top"] - lines[idx - 1]["bottom"]
            if gap > max(8.0, avg_height * 1.4):
                paragraphs.append(" ".join(current_para).strip())
                current_para = []
        current_para.append(line_text)

    if current_para:
        paragraphs.append(" ".join(current_para).strip())

    return "\n\n".join(paragraphs).strip()


def _score_text_candidate(text: str) -> float:
    """Heuristic quality score for competing text reconstructions."""
    if not text or not text.strip():
        return -1e9

    stripped = text.strip()
    lines = [line for line in stripped.splitlines() if line.strip()]
    words = _WORD_RE.findall(stripped)
    alnum = sum(1 for ch in stripped if ch.isalnum())
    one_word_lines = sum(1 for line in lines if len(line.split()) == 1)
    tiny_lines = sum(1 for line in lines if len(line) <= 2)
    weird_spacing = stripped.count("  ")

    return (
        alnum
        + len(words) * 2.0
        + len(lines) * 3.0
        - one_word_lines * 2.5
        - tiny_lines * 4.0
        - weird_spacing * 0.5
    )


def _looks_reasonable(text: str) -> bool:
    """Fast heuristic for deciding whether a page already has good text."""
    stripped = text.strip()
    if not stripped:
        return False

    words = _WORD_RE.findall(stripped)
    alnum = sum(1 for ch in stripped if ch.isalnum())

    return alnum >= 10 or len(words) >= 3 or len(stripped) >= 20


def _should_fast_accept_raw_text(text: str) -> bool:
    """Return True when raw PyMuPDF text is good enough to skip heavy reconstruction."""
    stripped = text.strip()
    if not _looks_reasonable(stripped):
        return False

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    words = _WORD_RE.findall(stripped)

    if len(lines) <= 6 and len(words) >= 5:
        return True

    return False


def _extract_pdfplumber_page_text(pdf_path: Path, page_number: int) -> str:
    """Optional high-fidelity fallback for pages that still look weak."""
    try:
        import pdfplumber
    except ImportError:
        return ""

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                return ""
            page = pdf.pages[page_number - 1]
            text = (
                page.extract_text(
                    layout=True,
                    x_tolerance=2,
                    y_tolerance=2,
                    keep_blank_chars=False,
                )
                or ""
            )
            return text.strip()
    except Exception:
        return ""


def _extract_page_text_from_bytes(pdf_bytes: bytes, page_number: int) -> str:
    """Extract text from a single page in a PDF byte stream as a last-resort fallback."""
    if not pdf_bytes:
        return ""

    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_number < 1 or page_number > len(doc):
            return ""
        page = doc[page_number - 1]

        chunks: List[str] = []
        contents = page.get_contents() or []
        if isinstance(contents, int):
            contents = [contents]

        for xref in contents:
            try:
                stream = doc.xref_stream(xref)
            except Exception:
                stream = None
            if not stream:
                continue
            for match in _STREAM_LITERAL_RE.finditer(stream):
                literal = match.group(0)[1:-1]
                try:
                    text = literal.decode("latin-1", errors="ignore")
                except Exception:
                    continue
                chunks.append(text.replace("\\n", "\n").replace("\\r", "\r").strip())

        chunks = [chunk for chunk in chunks if chunk]
        if chunks:
            return "\n".join(chunks).strip()

        if len(doc) == 1:
            return extract_text_from_pdf_bytes(pdf_bytes)

        return ""
    except Exception:
        return ""
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def extract_digital_page(
    page: fitz.Page,
    page_number: int,
    pdf_bytes: Optional[bytes] = None,
    pdf_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Extract and sort text from a single digital PDF page.
    """
    warnings: List[str] = []

    try:
        raw_text = page.get_text("text").strip()
    except Exception:
        raw_text = ""

    if _should_fast_accept_raw_text(raw_text):
        return {
            "text": raw_text,
            "blocks": [_build_synthetic_block(raw_text)],
            "warnings": warnings,
            "page_width": float(page.rect.width),
        }

    blocks = _extract_page_blocks(page)
    sorted_blocks = sort_digital_blocks(blocks) if blocks else []

    block_text = ""
    if sorted_blocks:
        paragraph_texts: List[str] = []
        current_para: List[str] = []
        prev_y1: Optional[float] = None

        for block in sorted_blocks:
            block_lines = [
                line.strip() for line in block["text"].splitlines() if line.strip()
            ]
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

        block_text = "\n\n".join(paragraph_texts).strip()
        block_text = "\n".join(merge_hyphenated_lines(block_text.splitlines())).strip()
    else:
        block_text = ""

    word_text = ""
    candidates = [text for text in [raw_text, block_text] if text.strip()]
    if candidates:
        best = max(candidates, key=_score_text_candidate)
        if _looks_reasonable(best):
            final_text = best
        else:
            word_text = _extract_words_text(page)
            candidates = [
                text for text in [raw_text, block_text, word_text] if text.strip()
            ]
            final_text = (
                max(candidates, key=_score_text_candidate) if candidates else ""
            )
    else:
        word_text = _extract_words_text(page)
        candidates = [
            text for text in [raw_text, block_text, word_text] if text.strip()
        ]
        final_text = max(candidates, key=_score_text_candidate) if candidates else ""

    if not final_text.strip() and pdf_path is not None:
        pdfplumber_text = _extract_pdfplumber_page_text(pdf_path, page_number)
        if pdfplumber_text.strip():
            candidates.append(pdfplumber_text)
            final_text = max(candidates, key=_score_text_candidate)

    if not final_text.strip() and pdf_bytes is not None:
        fallback_text = _extract_page_text_from_bytes(pdf_bytes, page_number)
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

    # M9 fix: always close the fitz document via try/finally
    try:
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
                result = extract_digital_page(
                    page,
                    page_num,
                    pdf_bytes=pdf_bytes,
                    pdf_path=pdf_path,
                )
            except Exception as exc:
                logger.error(f"Error extracting page {page_num}: {exc}", exc_info=True)
                result = {
                    "text": "",
                    "blocks": [],
                    "warnings": [f"Extraction failed: {exc}"],
                }

            result["page_number"] = page_num
            results.append(result)
            logger.debug(
                f"Extracted page {page_num}/{total} | chars={len(result['text'])}"
            )

    finally:
        doc.close()

    logger.info(
        f"Digital extraction complete | pages={len(results)} | file={pdf_path.name}"
    )
    return results
