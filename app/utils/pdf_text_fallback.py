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


def extract_text_fallback(pdf_bytes: bytes) -> list[dict]:
    """
    Minimal fallback: extract raw text from every page using best-effort
    parsing. Returns [{"page": int, "text": str}, ...].
    """
    if not pdf_bytes:
        return []

    results: list[dict] = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            results.append({"page": i + 1, "text": text.strip()})
        if any(item["text"] for item in results):
            return results
    except Exception as exc:
        logger.debug("pypdf fallback extraction failed: %s", exc)

    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        results = []
        for i, page in enumerate(doc):
            try:
                text = page.get_text("text").strip()
            except Exception:
                text = ""
            if not text:
                try:
                    blocks = page.get_text("blocks", sort=True) or []
                    texts = [
                        str(block[4]).strip()
                        for block in blocks
                        if len(block) >= 5 and str(block[4]).strip()
                    ]
                    text = "\n\n".join(texts).strip()
                except Exception:
                    text = ""
            results.append({"page": i + 1, "text": text})
        return results
    except Exception as exc:
        logger.debug("Fallback text extraction failed: %s", exc)
        return []
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def extract_text_from_pdf_bytes(
    pdf_bytes: bytes,
    page_number: Optional[int] = None,
) -> str:
    """Best-effort text extraction from a PDF byte stream."""
    pages = extract_text_fallback(pdf_bytes)
    if page_number is not None:
        if page_number < 1 or page_number > len(pages):
            return ""
        return (pages[page_number - 1].get("text") or "").strip()

    parts = [item.get("text", "").strip() for item in pages if item.get("text", "").strip()]
    return "\n\n---\n\n".join(parts).strip()
