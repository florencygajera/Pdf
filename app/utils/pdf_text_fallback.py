"""Fallback text extraction helpers for malformed PDFs.

These are intentionally conservative: they only run when the normal PyMuPDF
page-level extraction path fails or returns nothing useful.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import fitz

from app.utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf_bytes(pdf_bytes: bytes, page_number: Optional[int] = None) -> str:
    """Best-effort text extraction from a PDF byte stream."""
    if not pdf_bytes:
        return ""

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        if page_number is not None:
            if page_number < 1 or page_number > len(reader.pages):
                return ""
            try:
                text = reader.pages[page_number - 1].extract_text() or ""
                if text.strip():
                    return text.strip()
            except Exception:
                pass
        else:
            parts = []
            for page in reader.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n\n---\n\n".join(parts).strip()
    except Exception as exc:
        logger.debug("pypdf fallback extraction failed: %s", exc)

    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_number is not None:
            if page_number < 1 or page_number > len(doc):
                return ""
            page = doc[page_number - 1]
            try:
                text = page.get_text("text").strip()
                if text:
                    return text
            except Exception:
                pass
            try:
                blocks = page.get_text("blocks", sort=True) or []
                texts = [str(block[4]).strip() for block in blocks if len(block) >= 5 and str(block[4]).strip()]
                return "\n\n".join(texts).strip()
            except Exception:
                return ""

        parts = []
        for page in doc:
            try:
                text = page.get_text("text").strip()
                if text:
                    parts.append(text)
            except Exception:
                continue
        return "\n\n---\n\n".join(parts).strip()
    except Exception as exc:
        logger.debug("Fallback text extraction failed: %s", exc)
        return ""
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
