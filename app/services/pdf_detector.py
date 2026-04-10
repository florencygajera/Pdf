"""
PDF Type Detection Layer
Classifies each page (and the overall document) as:
  - digital  : has extractable text
  - scanned  : image-only, needs OCR
  - mixed    : document contains both kinds of pages

Uses PyMuPDF for fast text coverage analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF

from app.config.constants import (
    PDF_TYPE_DIGITAL,
    PDF_TYPE_MIXED,
    PDF_TYPE_SCANNED,
)
from app.config.settings import settings
from app.utils.logger import get_logger
from app.utils.pdf_text_fallback import extract_text_from_pdf_bytes

logger = get_logger(__name__)


@dataclass
class PageClassification:
    page_number: int  # 1-indexed
    pdf_type: str  # 'digital' | 'scanned'
    char_count: int
    has_images: bool
    text_coverage: float  # fraction of page area covered by text bboxes


@dataclass
class DocumentClassification:
    overall_type: str  # 'digital' | 'scanned' | 'mixed'
    pages: List[PageClassification]
    digital_page_count: int
    scanned_page_count: int
    total_pages: int


def _compute_text_coverage(page: fitz.Page) -> float:
    """
    Compute what fraction of the page area is covered by text bounding boxes.
    Returns a value in [0, 1].
    """
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    if page_area == 0:
        return 0.0

    text_area = 0.0
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)[
            "blocks"
        ]
    except Exception:
        return 0.0
    for block in blocks:
        if block.get("type") == 0:  # text block
            r = fitz.Rect(block["bbox"])
            text_area += r.width * r.height

    return min(text_area / page_area, 1.0)


def classify_page(
    page: fitz.Page,
    page_number: int,
    fallback_text: str = "",
) -> PageClassification:
    """
    Classify a single PDF page.
    A page is considered 'digital' if it has enough extractable characters.
    """
    try:
        raw_text = page.get_text("text")
    except Exception:
        raw_text = ""
    char_count = len(raw_text.strip())

    # Check for embedded images
    image_list = page.get_images(full=False)
    has_images = len(image_list) > 0

    text_coverage = _compute_text_coverage(page)

    if char_count < 5 and fallback_text.strip():
        char_count = max(char_count, len(fallback_text.strip()))

    # Decision: if char count is below threshold relative to page content
    is_digital = (
        char_count >= 5
        or text_coverage > settings.DIGITAL_TEXT_THRESHOLD
        or (char_count > 0 and not has_images)
    )

    pdf_type = PDF_TYPE_DIGITAL if is_digital else PDF_TYPE_SCANNED

    logger.debug(
        f"Page {page_number}: type={pdf_type}, chars={char_count}, "
        f"coverage={text_coverage:.3f}, images={len(image_list)}"
    )

    return PageClassification(
        page_number=page_number,
        pdf_type=pdf_type,
        char_count=char_count,
        has_images=has_images,
        text_coverage=text_coverage,
    )


def detect_pdf_type(pdf_path: Path) -> DocumentClassification:
    """
    Analyze all pages in a PDF and return the document classification.

    Raises:
        ValueError: If the file is encrypted or corrupt.
        FileNotFoundError: If the file doesn't exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except fitz.FileDataError as exc:
        raise ValueError(f"Corrupted PDF file: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if doc.is_encrypted:
        # Attempt empty-password decrypt (some PDFs are protected but readable)
        ok = doc.authenticate("")
        if not ok:
            raise ValueError("PDF is password-encrypted. Provide decryption password.")

    total_pages = len(doc)
    if total_pages == 0:
        raise ValueError("PDF has zero pages.")

    page_classifications: List[PageClassification] = []
    try:
        fallback_text = extract_text_from_pdf_bytes(pdf_path.read_bytes())
    except Exception:
        fallback_text = ""

    for i, page in enumerate(doc):
        pc = classify_page(page, page_number=i + 1, fallback_text=fallback_text)
        page_classifications.append(pc)

    doc.close()

    digital_count = sum(
        1 for p in page_classifications if p.pdf_type == PDF_TYPE_DIGITAL
    )
    scanned_count = total_pages - digital_count

    if digital_count == total_pages:
        overall = PDF_TYPE_DIGITAL
    elif scanned_count == total_pages:
        overall = PDF_TYPE_SCANNED
    else:
        overall = PDF_TYPE_MIXED

    logger.info(
        f"Document classification: {overall} | "
        f"total={total_pages}, digital={digital_count}, scanned={scanned_count}"
    )

    return DocumentClassification(
        overall_type=overall,
        pages=page_classifications,
        digital_page_count=digital_count,
        scanned_page_count=scanned_count,
        total_pages=total_pages,
    )
