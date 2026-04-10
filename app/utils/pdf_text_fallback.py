"""
Fallback text extraction for malformed PDFs.

PyMuPDF can fail on intentionally tiny or slightly malformed PDFs used in the
tests. This helper extracts visible text strings directly from PDF content
streams as a last-resort fallback.
"""

from __future__ import annotations

import re

_STREAM_RE = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_LITERAL_RE = re.compile(rb"\((?:\\.|[^\\)])*\)")


def _decode_pdf_literal(data: bytes) -> str:
    text = data.decode("latin-1", errors="ignore")
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        i += 1
        if i >= len(text):
            break

        nxt = text[i]
        escapes = {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "b": "\b",
            "f": "\f",
            "\\": "\\",
            "(": "(",
            ")": ")",
        }
        if nxt in escapes:
            out.append(escapes[nxt])
            i += 1
            continue

        if nxt.isdigit():
            oct_digits = nxt
            j = i + 1
            while j < len(text) and len(oct_digits) < 3 and text[j].isdigit():
                oct_digits += text[j]
                j += 1
            try:
                out.append(chr(int(oct_digits, 8)))
                i = j
                continue
            except ValueError:
                pass

        out.append(nxt)
        i += 1

    return "".join(out).strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""

    chunks = []
    for stream in _STREAM_RE.findall(pdf_bytes):
        literals = [
            _decode_pdf_literal(match.group(0)[1:-1])
            for match in _LITERAL_RE.finditer(stream)
        ]
        literals = [chunk for chunk in literals if chunk.strip()]
        if literals:
            chunks.extend(literals)

    if chunks:
        return "\n".join(chunks).strip()

    literals = [
        _decode_pdf_literal(match.group(0)[1:-1])
        for match in _LITERAL_RE.finditer(pdf_bytes)
        if _decode_pdf_literal(match.group(0)[1:-1]).strip()
    ]
    return "\n".join(literals).strip()
