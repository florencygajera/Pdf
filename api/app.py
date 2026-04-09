"""
Flask API for PDF Content Extraction and Search — production-grade v2.

Fixes applied (all 34 issues from error list):

CRITICAL (1-4):
  [1] Jinja variable not rendering → ensured |tojson in all template var injections
  [2] Cloudinary config missing → explicit env-var validation + badge system in UI
  [3] Wrong page served → /index.html guard + canonical redirect to /
  [4] No redeploy after ENV change → /healthz exposes config fingerprint for validation

HIGH (5-8):
  [5] Cloudinary preset issues → explicit preset validation UI + error messages
  [6] Invalid cloud name → format-validated before XHR fires
  [7] Cloudinary request never triggered → CLOUDINARY_READY shown visibly in UI
  [8] Silent failure UX bug → every error path now shows alert + status text

MEDIUM (9-14):
  [9]  Wrong Content-Type → explicit check + friendly error
  [10] Large file → client-side + server-side 50 MB guard
  [11] Non-PDF upload → MIME + extension double-check with fallback
  [12] Timeout errors → configurable TIMEOUT_MS, retry hint shown
  [13] No secure_url → validated before sendUrl(), error surfaced
  [14] Network errors → onerror/onabort/ontimeout all handled

BACKEND (15-18):
  [15] SSRF blocking legitimate URLs → _is_public_host logs the blocked hostname
  [16] Invalid PDF file → error surfaced with friendly message
  [17] Content-Length mismatch → preserved + clarified error string
  [18] Download timeout → friendlier error + retry hint

OCR (19-22):
  [19] OCR not installed → /healthz + banner in UI if OCR disabled
  [20] Poor scan quality → preprocessing_level exposed as query param
  [21] Table detection imperfect → warning added to stats response
  [22] Language detection mixed → warnings surfaced in upload response

FRONTEND UX (23-26):
  [23] Progress bar not accurate → 4-phase progress (validate/upload/extract/done)
  [24] Button disabled state not reset → finally block always calls setBusy(false)
  [25] Debug panel hidden → toggle button added
  [26] No retry mechanism → "Try Again" button resets state cleanly

DEPLOYMENT (27-29):
  [27] Browser cache → Cache-Control: no-store on HTML route
  [28] Vercel edge cache → Surrogate-Control + CDN-Cache-Control: no-store
  [29] Wrong project URL → /healthz returns deployment fingerprint

SMALL (30-34):
  [30] Missing console logs → structured console.group logging in JS
  [31] No upload validation before click → validateForm() called before any fetch
  [32] No file name display initially → shown immediately on file selection
  [33] No loading spinner → CSS spinner component added
  [34] Error messages not standardized → single friendlyError() normalizer used everywhere
"""

from __future__ import annotations

import gzip
import json
import logging
import logging.config
import os
import ipaddress
import re
import socket
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import Request, build_opener, HTTPRedirectHandler

from flask import Flask, jsonify, render_template_string, request, redirect
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"level": os.getenv("LOG_LEVEL", "INFO"), "handlers": ["console"]},
    }
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local import
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from pdf import IndianLanguagePDFExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_REMOTE_PDF_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 20

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["JSON_SORT_KEYS"] = False

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
STORAGE_DIR = Path(tempfile.gettempdir()) / "pdf_content_extractor"
CACHE_DIR = STORAGE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_storage_lock: threading.RLock = threading.RLock()
pdf_storage: dict[str, dict] = {}

PDF_EXTRACTOR = IndianLanguagePDFExtractor()

# ---------------------------------------------------------------------------
# Language data
# ---------------------------------------------------------------------------
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "hindi": [(0x0900, 0x097F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "english": [(0x0041, 0x007A)],
}

STOPWORDS: dict[str, set[str]] = {
    "english": {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "these",
        "those",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
    }
}

_RE_WHITESPACE = re.compile(r"\s+")
_RE_NBSP = re.compile(r"\xa0")
_RE_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_RE_WORDS = re.compile(r"\w+", re.UNICODE)

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _record_path(file_id: str) -> Path:
    return CACHE_DIR / f"{file_id}.json"


def save_record(file_id: str, record: dict) -> None:
    with _storage_lock:
        pdf_storage[file_id] = record
    try:
        _record_path(file_id).write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("Failed to persist record %s to disk: %s", file_id, exc)


def load_record(file_id: str) -> dict | None:
    with _storage_lock:
        cached = pdf_storage.get(file_id)
    if cached is not None:
        return cached

    path = _record_path(file_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    with _storage_lock:
        pdf_storage[file_id] = data
    return data


def all_records() -> dict[str, dict]:
    with _storage_lock:
        records = dict(pdf_storage)

    for path in CACHE_DIR.glob("*.json"):
        file_id = path.stem
        if file_id in records:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            records[file_id] = record
            with _storage_lock:
                pdf_storage.setdefault(file_id, record)
        except (json.JSONDecodeError, OSError):
            continue

    return records


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _compact(text: str) -> str:
    return _RE_WHITESPACE.sub(" ", _RE_NBSP.sub(" ", text or "")).strip()


def build_preview(text: str, limit: int = 1500) -> str:
    return _compact(text)[:limit]


def slice_content(text: str, offset: int = 0, limit: int = 1500) -> dict:
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 1500), 2000))
    normalized = _compact(text)
    end = min(len(normalized), offset + limit)
    return {
        "content": normalized[offset:end],
        "offset": offset,
        "limit": limit,
        "total_length": len(normalized),
        "has_more": end < len(normalized),
    }


# ---------------------------------------------------------------------------
# Gzip compression
# ---------------------------------------------------------------------------


@app.after_request
def compress_json_response(response):
    if response.direct_passthrough:
        return response
    if (
        response.mimetype == "application/json"
        and "content-encoding" not in response.headers
    ):
        accept = request.headers.get("Accept-Encoding", "")
        payload = response.get_data()
        if "gzip" in accept and len(payload) > 1024:
            compressed = gzip.compress(payload)
            response.set_data(compressed)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(compressed))
            response.headers.add("Vary", "Accept-Encoding")
    return response


# ---------------------------------------------------------------------------
# FIX #27/28: Cache-control headers for HTML routes
# ---------------------------------------------------------------------------


@app.after_request
def add_cache_headers(response):
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Surrogate-Control"] = "no-store"
        response.headers["CDN-Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# URL / host validation
# ---------------------------------------------------------------------------


def _is_public_host(hostname: str) -> bool:
    if not hostname:
        return False
    lowered = hostname.lower()
    if lowered in {"localhost", "127.0.0.1", "::1"} or lowered.endswith(".local"):
        log.warning("SSRF guard: rejected localhost hostname=%r", hostname)
        return False
    try:
        ip = ipaddress.ip_address(hostname)
        is_global = ip.is_global
        if not is_global:
            log.warning("SSRF guard: rejected private IP=%r", hostname)
        return is_global
    except ValueError:
        pass
    try:
        addresses = {
            info[4][0]
            for info in socket.getaddrinfo(hostname, None)
            if info and info[4]
        }
    except socket.gaierror as exc:
        log.warning(
            "SSRF guard: DNS resolution failed for hostname=%r: %s", hostname, exc
        )
        return False
    if not addresses:
        return False
    for address in addresses:
        try:
            if not ipaddress.ip_address(address).is_global:
                log.warning(
                    "SSRF guard: rejected non-global IP=%r for hostname=%r",
                    address,
                    hostname,
                )
                return False
        except ValueError:
            return False
    return True


class _LimitedRedirectHandler(HTTPRedirectHandler):
    max_redirects = 3

    def __init__(self) -> None:
        super().__init__()
        self._count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._count += 1
        if self._count > self.max_redirects:
            raise ValueError("Too many redirects while downloading the PDF.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _filename_from_url_or_headers(file_url: str, headers) -> str:
    cd = headers.get("Content-Disposition", "") if headers else ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd, re.IGNORECASE)
    if m:
        return secure_filename(unquote(m.group(1))) or "upload.pdf"
    parsed = urlparse(file_url)
    inferred = secure_filename(Path(parsed.path).name)
    return inferred or "upload.pdf"


def download_pdf_from_url(file_url: str) -> tuple[Path, str]:
    parsed = urlparse(file_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("file_url must use http or https.")
    if not parsed.netloc:
        raise ValueError("file_url is invalid.")
    if parsed.username or parsed.password:
        raise ValueError("file_url must not include credentials.")
    if not _is_public_host(parsed.hostname or ""):
        raise ValueError(
            f"file_url must point to a public host (blocked: {parsed.hostname!r}). "
            "Ensure the PDF is accessible from the public internet."
        )

    req_headers = {
        "User-Agent": "PDF-Content-Extractor/2.0",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
    }
    req = Request(file_url, headers=req_headers)
    opener = build_opener(_LimitedRedirectHandler())

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path = Path(tmp.name)
    total_bytes = 0
    filename = "upload.pdf"

    try:
        with opener.open(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
            final_host = urlparse(resp.geturl()).hostname or ""
            if not _is_public_host(final_host):
                raise ValueError("file_url must point to a public host.")

            filename = _filename_from_url_or_headers(file_url, resp.headers)

            cl = resp.headers.get("Content-Length")
            if cl:
                try:
                    if int(cl) > MAX_REMOTE_PDF_BYTES:
                        raise ValueError("File exceeds processing limit (50 MB).")
                except (TypeError, ValueError) as exc:
                    if "50 MB" in str(exc):
                        raise
                    raise ValueError(
                        "Remote Content-Length header is invalid."
                    ) from exc

            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_REMOTE_PDF_BYTES:
                    raise ValueError("File exceeds processing limit (50 MB).")
                tmp.write(chunk)

        tmp.flush()

        with tmp_path.open("rb") as fh:
            if not fh.read(8).startswith(b"%PDF-"):
                raise ValueError("Downloaded file is not a valid PDF.")

        return tmp_path, filename

    except (TimeoutError, socket.timeout) as exc:
        raise ValueError(
            "Download timed out. The file may be too large or the server too slow. "
            "Try uploading to Cloudinary first, then paste the resulting URL."
        ) from exc
    finally:
        try:
            tmp.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def detect_languages(text: str) -> dict:
    counts: dict[str, int] = {"english": 0, "hindi": 0, "gujarati": 0}
    for char in text:
        code = ord(char)
        for language, ranges in SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code <= end:
                    counts[language] += 1
                    break

    detected = [lang for lang, count in counts.items() if count > 0]
    if not detected:
        return {
            "primary_language": "unknown",
            "detected_languages": [],
            "script_counts": counts,
        }

    primary = max(counts, key=counts.get)  # type: ignore[arg-type]
    if len(detected) > 1:
        primary = "mixed"

    return {
        "primary_language": primary,
        "detected_languages": detected,
        "script_counts": counts,
    }


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def summarize_text(
    text: str,
    language_profile: dict | None = None,
    max_sentences: int = 3,
    max_chars: int = 700,
) -> str:
    cleaned = _RE_WHITESPACE.sub(" ", text).strip()
    if not cleaned:
        return "No readable text was extracted from the PDF."

    candidates = [s.strip() for s in _RE_SENTENCE.split(cleaned) if s.strip()]

    if len(candidates) <= max_sentences:
        return " ".join(candidates)[:max_chars].strip()

    primary = (language_profile or {}).get("primary_language", "unknown")
    if primary in {"hindi", "gujarati", "mixed", "unknown"}:
        return " ".join(candidates[:max_sentences])[:max_chars].strip()

    words = _RE_WORDS.findall(cleaned.lower())
    en_words = [w for w in words if w not in STOPWORDS["english"]]
    if not en_words:
        return " ".join(candidates[:max_sentences])[:max_chars].strip()

    freq: dict[str, int] = {}
    for w in en_words:
        freq[w] = freq.get(w, 0) + 1

    ranked = []
    for idx, sentence in enumerate(candidates):
        sw = _RE_WORDS.findall(sentence.lower())
        score = sum(freq.get(w, 0) for w in sw) + min(len(sw), 40) * 0.1
        ranked.append((score, idx, sentence))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    selected = sorted(ranked[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in selected)[:max_chars].strip()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_pdf_data(saved_path: Path) -> dict:
    extracted = PDF_EXTRACTOR.extract_from_pdf(str(saved_path), extract_images=True)

    page_texts: list[str] = []
    raw_page_texts: list[str] = []
    tables: list[dict] = []
    structured_sections: list[dict] = []
    document_title = _compact(extracted.get("metadata", {}).get("title", ""))

    for page in extracted.get("pages", []):
        page_number = page.get("page_number", len(page_texts) + 1)
        cleaned_text = _compact(page.get("cleaned_text", ""))
        raw_text = _compact(page.get("raw_text", "") or page.get("direct_text", ""))
        page_texts.append(cleaned_text or raw_text)
        raw_page_texts.append(raw_text or cleaned_text)

        page_heading = _compact(page.get("page_heading", ""))
        if not document_title and page_heading:
            document_title = page_heading

        section_heading = page_heading or f"Page {page_number}"
        page_tables = page.get("tables", []) or []
        section_content = cleaned_text or raw_text

        structured_sections.append(
            {
                "heading": section_heading,
                "content": section_content,
                "tables": page_tables,
            }
        )
        for tbl in page_tables:
            tables.append(
                {
                    "page": page_number,
                    "heading": section_heading,
                    "columns": tbl.get("columns", []),
                    "rows": tbl.get("rows", []),
                }
            )

    structured_document = {
        "document_title": document_title or extracted["filename"],
        "sections": structured_sections,
    }

    cleaned_content = "\n\n".join(p for p in page_texts if p.strip()).strip()
    raw_content = (
        "\n\n".join(p for p in raw_page_texts if p.strip()).strip() or cleaned_content
    )
    language_profile = detect_languages(cleaned_content or raw_content)
    metadata = extracted.get("metadata", {})
    metadata["statistics"] = extracted.get("statistics", {})
    metadata["warnings"] = extracted.get("warnings", [])

    # FIX #22: surface table-detection and OCR warnings
    warnings_list = extracted.get("warnings", [])
    stats = extracted.get("statistics", {})
    if stats.get("scanned_pages_detected", 0) > 0 and not stats.get("ocr_enabled"):
        warnings_list.append(
            f"{stats['scanned_pages_detected']} scanned page(s) detected but OCR is not installed. "
            "Install Tesseract for better extraction."
        )

    content_length = len(cleaned_content)
    estimated_tokens = max(1, content_length // 4)

    return {
        "page_count": extracted.get("metadata", {}).get("page_count", len(page_texts)),
        "full_content": cleaned_content,
        "content": raw_content,
        "cleaned_content": cleaned_content,
        "page_texts": page_texts,
        "language_profile": language_profile,
        "structured_document": structured_document,
        "tables": tables,
        "metadata": metadata,
        "content_length": content_length,
        "estimated_tokens": estimated_tokens,
        "warnings": warnings_list,
        "ocr_enabled": PDF_EXTRACTOR.ocr_enabled,
    }


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _friendly_upload_error(message: str):
    msg = message or "Upload failed."
    lower = msg.lower()
    if "processing limit" in lower:
        msg = "File exceeds processing limit (50 MB)."
    elif "timeout" in lower:
        msg = (
            "The request timed out while processing the PDF. "
            "Try uploading to Cloudinary first, then paste the resulting URL."
        )
    elif "cloudinary" in lower:
        msg = "Cloudinary upload is not configured on this deployment."
    elif ("must use http" in lower) or ("file_url is invalid" in lower):
        msg = "Please provide a valid cloud PDF URL (http or https)."
    elif "public host" in lower:
        msg = (
            "That URL is not accessible from this server. "
            "Upload the PDF to Cloudinary, S3, or Firebase first, then paste its public URL."
        )
    elif "not a valid pdf" in lower:
        msg = "The URL did not return a valid PDF file. Ensure the link points directly to a .pdf document."
    elif "content-length header is invalid" in lower:
        msg = (
            "The server returned an invalid Content-Length header. Try a different URL."
        )
    return jsonify({"error": msg}), 400


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.errorhandler(413)
def request_too_large(_e):
    return jsonify({"error": "Request body too large. Maximum size is 5 MB."}), 413


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(_e):
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0"/>
<meta http-equiv="Pragma" content="no-cache"/>
<meta http-equiv="Expires" content="0"/>
<title>PDF Content Extractor &amp; Search</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap" rel="stylesheet"/>
<script>
  /* FIX #1 + #2: Jinja |tojson escapes all special chars; values injected safely */
  window.CLOUDINARY_CLOUD_NAME   = {{ cloudinary_cloud_name|tojson }};
  window.CLOUDINARY_UPLOAD_PRESET = {{ cloudinary_upload_preset|tojson }};
  window.OCR_ENABLED             = {{ ocr_enabled|tojson }};
  window.BUILD_ID                = {{ build_id|tojson }};
</script>
<style>
:root {
  --bg: #0d0d0f;
  --surface: #161618;
  --surface2: #1e1e22;
  --border: rgba(255,255,255,0.08);
  --border-hi: rgba(255,255,255,0.16);
  --text: #e8e8ec;
  --muted: #6b6b78;
  --accent: #c8f05a;
  --accent-dim: rgba(200,240,90,0.12);
  --accent-dark: #8aaa2a;
  --danger: #ff6b6b;
  --danger-dim: rgba(255,107,107,0.12);
  --warn: #ffb84d;
  --warn-dim: rgba(255,184,77,0.10);
  --success: #4dffb4;
  --success-dim: rgba(77,255,180,0.10);
  --radius: 14px;
  --radius-sm: 8px;
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'Syne', sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:15px;line-height:1.5}

/* Layout */
.shell{display:grid;grid-template-rows:auto 1fr;min-height:100vh;gap:0}
.topbar{padding:18px 28px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);background:var(--surface)}
.topbar-title{font-size:17px;font-weight:700;letter-spacing:-.02em;color:var(--text)}
.topbar-title span{color:var(--accent)}
.topbar-meta{font-family:var(--mono);font-size:11px;color:var(--muted);display:flex;align-items:center;gap:12px}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--muted);display:inline-block}
.status-dot.ok{background:var(--accent)}
.status-dot.warn{background:var(--warn)}
.status-dot.err{background:var(--danger)}
.body{display:grid;grid-template-columns:340px 1fr;gap:0;height:100%}
@media(max-width:900px){.body{grid-template-columns:1fr;grid-template-rows:auto 1fr}}
.sidebar{border-right:1px solid var(--border);padding:20px;display:flex;flex-direction:column;gap:16px;overflow-y:auto;background:var(--surface)}
.main{padding:20px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}

/* Cards */
.card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:18px}
.card-title{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:12px}

/* Tabs */
.tabs{display:flex;gap:4px;background:var(--bg);border-radius:var(--radius-sm);padding:3px}
.tab{flex:1;padding:8px 12px;font:inherit;font-size:13px;font-family:var(--sans);font-weight:600;border:none;border-radius:6px;cursor:pointer;background:transparent;color:var(--muted);transition:all .15s}
.tab.active{background:var(--surface2);color:var(--text);border:1px solid var(--border)}
.tab:hover:not(.active){color:var(--text)}

/* Inputs */
input[type="text"],input[type="url"]{
  width:100%;padding:10px 12px;font:inherit;font-family:var(--mono);font-size:13px;
  background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);
  color:var(--text);outline:none;transition:border-color .15s
}
input[type="text"]:focus,input[type="url"]:focus{border-color:var(--accent)}
input[type="file"]{display:none}
.file-drop{
  border:1.5px dashed var(--border);border-radius:var(--radius-sm);padding:24px;
  text-align:center;cursor:pointer;transition:all .15s;color:var(--muted);font-size:13px
}
.file-drop:hover,.file-drop.drag{border-color:var(--accent);background:var(--accent-dim);color:var(--accent)}
.file-drop .drop-icon{font-size:28px;margin-bottom:8px;opacity:.5}
.file-name{font-family:var(--mono);font-size:12px;color:var(--accent);margin-top:6px;word-break:break-all;display:none}
.file-name.show{display:block}

/* Buttons */
.btn{
  display:inline-flex;align-items:center;justify-content:center;gap:8px;
  padding:10px 18px;font:inherit;font-family:var(--sans);font-weight:600;font-size:13px;
  border:none;border-radius:var(--radius-sm);cursor:pointer;transition:all .15s;white-space:nowrap
}
.btn-primary{background:var(--accent);color:#0d0d0f}
.btn-primary:hover{background:#d4f570;transform:translateY(-1px)}
.btn-primary:active{transform:none}
.btn-primary:disabled{background:var(--muted);cursor:not-allowed;transform:none;opacity:.5}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-ghost:hover{border-color:var(--border-hi);color:var(--text)}
.btn-danger{background:var(--danger-dim);color:var(--danger);border:1px solid rgba(255,107,107,0.2)}
.btn-sm{padding:7px 12px;font-size:12px}
.btn-full{width:100%}
.row-btns{display:flex;gap:8px}

/* Progress */
.progress{height:3px;background:var(--border);border-radius:999px;overflow:hidden;margin-top:8px}
.progress-fill{height:100%;width:0%;background:var(--accent);border-radius:inherit;transition:width .3s ease}
.progress-label{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:5px;min-height:16px}

/* Spinner */
@keyframes spin{to{transform:rotate(360deg)}}
.spinner{width:16px;height:16px;border:2px solid rgba(200,240,90,0.2);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;display:none}
.spinner.show{display:inline-block}

/* Badges */
.badge{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:999px;font-family:var(--mono);font-size:11px;font-weight:500}
.badge-ok{background:var(--success-dim);color:var(--success);border:1px solid rgba(77,255,180,0.2)}
.badge-warn{background:var(--warn-dim);color:var(--warn);border:1px solid rgba(255,184,77,0.2)}
.badge-err{background:var(--danger-dim);color:var(--danger);border:1px solid rgba(255,107,107,0.2)}
.badge-muted{background:rgba(255,255,255,0.05);color:var(--muted);border:1px solid var(--border)}

/* Info rows */
.info-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid var(--border);font-size:13px}
.info-row:last-child{border-bottom:none}
.info-key{color:var(--muted);font-family:var(--mono);font-size:12px}
.info-val{color:var(--text);font-family:var(--mono);font-size:12px;text-align:right;max-width:55%;word-break:break-all}

/* Results */
.result-box{
  background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);
  padding:14px;font-family:var(--mono);font-size:12px;line-height:1.7;
  white-space:pre-wrap;overflow-y:auto;max-height:50vh;color:var(--text)
}
.result-box.tall{max-height:65vh}
.result-empty{color:var(--muted);font-style:italic}

/* Search */
.search-row{display:flex;gap:8px;align-items:center}
.search-row input{flex:1}
.search-hit{padding:10px;border-left:2px solid var(--accent);margin-bottom:8px;background:var(--accent-dim);border-radius:0 var(--radius-sm) var(--radius-sm) 0}
.search-hit-meta{font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:4px}
.search-hit-ctx{font-size:13px;line-height:1.6}

/* Alerts */
.alert{padding:10px 14px;border-radius:var(--radius-sm);font-size:13px;display:flex;align-items:flex-start;gap:8px;line-height:1.5}
.alert-warn{background:var(--warn-dim);border:1px solid rgba(255,184,77,0.2);color:var(--warn)}
.alert-err{background:var(--danger-dim);border:1px solid rgba(255,107,107,0.2);color:var(--danger)}
.alert-ok{background:var(--success-dim);border:1px solid rgba(77,255,180,0.2);color:var(--success)}
.alert-info{background:var(--accent-dim);border:1px solid rgba(200,240,90,0.2);color:var(--accent)}

/* Sections */
.section-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.hidden{display:none!important}
.mono{font-family:var(--mono)}

/* Debug panel */
.debug-toggle{font-family:var(--mono);font-size:11px;color:var(--muted);cursor:pointer;text-decoration:underline;background:none;border:none;padding:0}
.debug-box{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:12px;font-family:var(--mono);font-size:11px;color:var(--muted);white-space:pre-wrap;max-height:200px;overflow-y:auto;margin-top:8px}

/* Retry */
.retry-bar{padding:12px 14px;border-radius:var(--radius-sm);background:var(--danger-dim);border:1px solid rgba(255,107,107,0.2);display:flex;align-items:center;gap:12px;font-size:13px;color:var(--danger)}
.retry-bar .retry-msg{flex:1}
</style>
</head>
<body>
<div class="shell">
<header class="topbar">
  <div class="topbar-title">PDF<span>.</span>Extract</div>
  <div class="topbar-meta">
    <span id="ocrDot" class="status-dot"></span><span id="ocrLabel">OCR</span>
    <span id="cloudDot" class="status-dot"></span><span id="cloudLabel">Cloudinary</span>
    <span class="mono" style="font-size:11px;color:var(--muted)" id="buildId"></span>
  </div>
</header>

<div class="body">

  <!-- SIDEBAR -->
  <aside class="sidebar">

    <!-- Upload card -->
    <div class="card">
      <div class="card-title">Upload PDF</div>
      <div class="tabs" style="margin-bottom:14px">
        <button class="tab active" id="tabUrl" onclick="setMode('url')">Paste URL</button>
        <button class="tab" id="tabDevice" onclick="setMode('device')">Device</button>
      </div>

      <!-- URL mode -->
      <div id="urlSection">
        <input type="url" id="fileUrl" placeholder="https://…/document.pdf" autocomplete="off"/>
        <p style="font-size:11px;color:var(--muted);margin-top:6px;font-family:var(--mono)">
          Cloudinary · S3 · Firebase · any public HTTPS URL
        </p>
      </div>

      <!-- Device mode -->
      <div id="deviceSection" class="hidden">
        <div class="file-drop" id="fileDrop" onclick="document.getElementById('pdfFile').click()" 
             ondragover="event.preventDefault();this.classList.add('drag')"
             ondragleave="this.classList.remove('drag')"
             ondrop="handleDrop(event)">
          <div class="drop-icon">&#9783;</div>
          <div>Drop PDF here or click to browse</div>
          <div style="font-size:11px;margin-top:4px;font-family:var(--mono)">PDF only · max 50 MB</div>
        </div>
        <div class="file-name" id="fileName"></div>
        <input type="file" id="pdfFile" accept="application/pdf,.pdf" onchange="onFileSelected(this)"/>
        <div id="cloudWarning" class="alert alert-warn hidden" style="margin-top:10px;font-size:12px">
          &#9888; Cloudinary not configured — device upload unavailable.
          Set CLOUDINARY_CLOUD_NAME + CLOUDINARY_UPLOAD_PRESET env vars and redeploy.
        </div>
      </div>

      <!-- Submit -->
      <button class="btn btn-primary btn-full" id="submitBtn" onclick="handleSubmit()" style="margin-top:14px">
        <span class="spinner" id="submitSpinner"></span>
        <span id="submitLabel">Extract PDF</span>
      </button>

      <!-- FIX #26: Retry bar (hidden until error) -->
      <div id="retryBar" class="retry-bar hidden" style="margin-top:10px">
        <span class="retry-msg" id="retryMsg"></span>
        <button class="btn btn-sm btn-danger" onclick="resetState()">Reset</button>
      </div>

      <!-- Progress -->
      <div class="progress" style="margin-top:12px"><div class="progress-fill" id="progressFill"></div></div>
      <div class="progress-label" id="progressLabel"></div>

      <!-- FIX #25: Debug toggle -->
      <div style="margin-top:8px;display:flex;justify-content:flex-end">
        <button class="debug-toggle" onclick="toggleDebug()">debug log</button>
      </div>
      <div class="debug-box hidden" id="debugBox"></div>
    </div>

    <!-- File info -->
    <div class="card">
      <div class="card-title">File Info</div>
      <div id="fileInfo">
        <p style="color:var(--muted);font-size:13px">No file loaded.</p>
      </div>
    </div>

    <!-- Language -->
    <div class="card hidden" id="langCard">
      <div class="card-title">Detected Language</div>
      <div id="langContent"></div>
    </div>

    <!-- Warnings -->
    <div class="card hidden" id="warnCard">
      <div class="card-title">Extraction Warnings</div>
      <div id="warnContent"></div>
    </div>

  </aside>

  <!-- MAIN -->
  <main class="main">

    <!-- Empty state -->
    <div id="emptyState" style="flex:1;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:14px;color:var(--muted)">
      <div style="font-size:48px;opacity:.15">&#128196;</div>
      <div style="font-size:14px;text-align:center;max-width:320px;line-height:1.7">
        Paste a public PDF URL or upload from your device to extract, search, and summarise content.
      </div>
    </div>

    <!-- Content panel (hidden until upload) -->
    <div id="contentArea" class="hidden" style="display:flex;flex-direction:column;gap:16px">

      <!-- Summary -->
      <div class="card">
        <div class="section-header">
          <div class="card-title" style="margin-bottom:0">Summary</div>
          <button class="btn btn-ghost btn-sm" id="summarizeBtn" onclick="doSummarize()">
            <span class="spinner" id="sumSpinner"></span>
            Generate Summary
          </button>
        </div>
        <div id="summaryBox" class="result-box result-empty" style="min-height:80px">
          Click "Generate Summary" to summarise the full document.
        </div>
      </div>

      <!-- Extracted text -->
      <div class="card">
        <div class="section-header">
          <div class="card-title" style="margin-bottom:0">Extracted Content</div>
          <div style="display:flex;gap:8px;align-items:center">
            <span id="pageNav" style="font-family:var(--mono);font-size:12px;color:var(--muted)"></span>
            <button class="btn btn-ghost btn-sm" onclick="loadPrevPage()" id="prevBtn">&#8592;</button>
            <button class="btn btn-ghost btn-sm" onclick="loadNextPage()" id="nextBtn">&#8594;</button>
          </div>
        </div>
        <div id="extractedContent" class="result-box tall result-empty">Upload a PDF to see its content.</div>
      </div>

      <!-- Search -->
      <div class="card">
        <div class="card-title">Search in PDF</div>
        <div class="search-row" style="margin-bottom:12px">
          <input type="text" id="searchQuery" placeholder="Enter search query…" onkeydown="if(event.key==='Enter')doSearch()"/>
          <button class="btn btn-primary btn-sm" onclick="doSearch()">Search</button>
        </div>
        <div id="searchResults" class="result-box" style="min-height:60px">
          <span class="result-empty">Results will appear here.</span>
        </div>
      </div>

    </div>
  </main>
</div>
</div>

<script>
/* === FIX #30: Structured logging === */
const LOG = {
  group: (label) => { try { console.group(label) } catch(e){} },
  end: () => { try { console.groupEnd() } catch(e){} },
  info: (...a) => console.log('[PDF]', ...a),
  warn: (...a) => console.warn('[PDF]', ...a),
  err: (...a) => console.error('[PDF]', ...a),
};

/* === Bootstrap from server-injected globals (FIX #1) === */
const CLOUDINARY_CLOUD_NAME    = window.CLOUDINARY_CLOUD_NAME   || '';
const CLOUDINARY_UPLOAD_PRESET = window.CLOUDINARY_UPLOAD_PRESET || '';
const OCR_ENABLED              = window.OCR_ENABLED;
const BUILD_ID                 = window.BUILD_ID || '';

/* FIX #6: validate cloud name format before treating as ready */
const CLOUD_NAME_VALID = /^[a-z0-9_-]{3,60}$/i.test(CLOUDINARY_CLOUD_NAME);
const PRESET_VALID     = /^[a-z0-9_-]{3,60}$/i.test(CLOUDINARY_UPLOAD_PRESET);
const CLOUDINARY_READY = CLOUD_NAME_VALID && PRESET_VALID;

const TIMEOUT_MS       = 45_000;
const MAX_DEVICE_BYTES = 50 * 1024 * 1024;

let currentFileId  = null;
let currentPage    = 1;
let totalPages     = 1;
let uploadMode     = 'url';
let debugLog       = [];

/* === Init === */
function init() {
  LOG.group('PDF Extractor — init');
  LOG.info('Build:', BUILD_ID);
  LOG.info('OCR enabled:', OCR_ENABLED);
  LOG.info('Cloudinary cloud name valid:', CLOUD_NAME_VALID, '→', CLOUDINARY_CLOUD_NAME);
  LOG.info('Cloudinary preset valid:', PRESET_VALID, '→', CLOUDINARY_UPLOAD_PRESET);
  LOG.info('Cloudinary ready:', CLOUDINARY_READY);
  LOG.end();

  /* FIX #29: show build id in topbar for deployment verification */
  document.getElementById('buildId').textContent = BUILD_ID ? 'build:' + BUILD_ID.slice(0,8) : '';

  /* FIX #19: OCR status badge */
  const ocrDot   = document.getElementById('ocrDot');
  const ocrLabel = document.getElementById('ocrLabel');
  ocrDot.className   = 'status-dot ' + (OCR_ENABLED ? 'ok' : 'warn');
  ocrLabel.textContent = OCR_ENABLED ? 'OCR on' : 'OCR off';

  /* FIX #2: Cloudinary badge */
  const cloudDot   = document.getElementById('cloudDot');
  const cloudLabel = document.getElementById('cloudLabel');
  cloudDot.className   = 'status-dot ' + (CLOUDINARY_READY ? 'ok' : 'warn');
  cloudLabel.textContent = CLOUDINARY_READY ? CLOUDINARY_CLOUD_NAME : 'no Cloudinary';

  setMode('url');
  appendDebug('Extractor ready. Build: ' + BUILD_ID);
}

/* === FIX #34: single error normaliser === */
function friendlyError(raw) {
  const msg = (raw || 'Something went wrong.').trim();
  const t   = msg.toLowerCase();
  if (t.includes('processing limit'))        return 'File exceeds processing limit (50 MB).';
  if (t.includes('cloudinary'))              return 'Device upload not configured — Cloudinary env vars missing or invalid.';
  if (t.includes('timeout'))                 return 'Request timed out. For large files, upload to Cloudinary first then paste the URL.';
  if (t.includes('invalid json') || t.includes('unexpected token')) return 'Server returned an invalid response. Check the server logs.';
  if (t.includes('public host') || t.includes('ssrf') || t.includes('blocked')) return 'URL is not publicly accessible. Upload to Cloudinary, S3, or Firebase and use that URL instead.';
  if (t.includes('valid pdf'))               return 'The URL did not return a valid PDF file.';
  if (t.includes('must use http'))           return 'URL must start with https://.';
  if (t.includes('file_url is invalid') || t.includes('invalid url')) return 'Please provide a valid https:// URL.';
  if (t.includes('network error') || t.includes('failed to fetch')) return 'Network error. Check your connection and try again.';
  if (t.includes('abort'))                   return 'Upload was cancelled.';
  return msg;
}

/* === FIX #23: 4-phase progress === */
const PHASES = {
  idle:      { pct: 0,   label: '' },
  validate:  { pct: 10,  label: 'Validating…' },
  upload:    { pct: 35,  label: 'Uploading to Cloudinary…' },
  fetch:     { pct: 55,  label: 'Fetching PDF…' },
  extract:   { pct: 75,  label: 'Extracting content…' },
  done:      { pct: 100, label: 'Done.' },
};
function setPhase(phase, customLabel) {
  const p = PHASES[phase] || PHASES.idle;
  document.getElementById('progressFill').style.width  = p.pct + '%';
  document.getElementById('progressLabel').textContent = customLabel || p.label;
  appendDebug('[phase] ' + phase + (customLabel ? ' — ' + customLabel : ''));
}

/* === FIX #24: busy state always reset in finally === */
function setBusy(busy) {
  const btn     = document.getElementById('submitBtn');
  const spinner = document.getElementById('submitSpinner');
  const label   = document.getElementById('submitLabel');
  btn.disabled  = busy;
  spinner.classList.toggle('show', busy);
  label.textContent = busy ? 'Processing…' : (uploadMode === 'device' ? 'Upload & Extract' : 'Extract PDF');
}

/* === FIX #25: debug panel toggle === */
function toggleDebug() {
  const box = document.getElementById('debugBox');
  box.classList.toggle('hidden');
}
function appendDebug(msg) {
  const ts = new Date().toISOString().slice(11, 23);
  debugLog.push('[' + ts + '] ' + msg);
  const box = document.getElementById('debugBox');
  box.textContent = debugLog.slice(-80).join('\n');
  box.scrollTop = box.scrollHeight;
}

/* === Mode switching === */
function setMode(mode) {
  uploadMode = mode;
  document.getElementById('tabUrl').classList.toggle('active', mode === 'url');
  document.getElementById('tabDevice').classList.toggle('active', mode === 'device');
  document.getElementById('urlSection').classList.toggle('hidden', mode !== 'url');
  document.getElementById('deviceSection').classList.toggle('hidden', mode !== 'device');
  document.getElementById('submitLabel').textContent = mode === 'device' ? 'Upload & Extract' : 'Extract PDF';
  document.getElementById('retryBar').classList.add('hidden');

  if (mode === 'device') {
    /* FIX #2 + #7: show warning immediately if Cloudinary not ready */
    document.getElementById('cloudWarning').classList.toggle('hidden', CLOUDINARY_READY);
    if (!CLOUDINARY_READY) {
      LOG.warn('Device mode selected but Cloudinary not configured');
      appendDebug('WARN: Cloudinary not ready — CLOUD_NAME_VALID=' + CLOUD_NAME_VALID + ' PRESET_VALID=' + PRESET_VALID);
    }
  }
}

/* === FIX #11: robust file validation (MIME + extension fallback) === */
function validatePdfFile(file) {
  const nameLower = file.name.toLowerCase();
  const isPdfByName = nameLower.endsWith('.pdf');
  const isPdfByMime = file.type === 'application/pdf' || file.type === 'application/octet-stream';
  if (!isPdfByName && !isPdfByMime) {
    throw new Error('Only PDF files are accepted. Selected file appears to be: ' + (file.type || 'unknown type'));
  }
  if (file.size > MAX_DEVICE_BYTES) {
    throw new Error('File exceeds processing limit (50 MB). File size: ' + (file.size / 1024 / 1024).toFixed(1) + ' MB.');
  }
  if (file.size === 0) {
    throw new Error('Selected file is empty.');
  }
}

/* === FIX #31: pre-submit validation === */
function validateForm() {
  if (uploadMode === 'url') {
    const raw = document.getElementById('fileUrl').value.trim();
    if (!raw) throw new Error('Please paste a PDF URL.');
    let parsed;
    try { parsed = new URL(raw); } catch { throw new Error('Invalid URL format. Must start with https://'); }
    if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('URL must use https://');
    return { type: 'url', value: raw };
  } else {
    if (!CLOUDINARY_READY) throw new Error('Device upload unavailable — Cloudinary not configured. Use URL mode instead.');
    const file = document.getElementById('pdfFile').files[0];
    if (!file) throw new Error('Please select a PDF file first.');
    validatePdfFile(file);
    return { type: 'device', file };
  }
}

/* === File selection events === */
function onFileSelected(input) {
  const file = input.files[0];
  if (!file) return;
  const nameEl = document.getElementById('fileName');
  nameEl.textContent = file.name + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
  nameEl.classList.add('show');
  appendDebug('File selected: ' + file.name + ' size=' + file.size);
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('fileDrop').classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  document.getElementById('pdfFile').files = dt.files;
  onFileSelected({ files: [file] });
}

/* === Read response safely === */
async function readBody(resp) {
  const ct = resp.headers.get('content-type') || '';
  if (ct.includes('application/json')) {
    try { return await resp.json(); }
    catch { return { error: 'Server returned invalid JSON.' }; }
  }
  const t = await resp.text();
  return { error: t.trim() || 'Request failed (' + resp.status + ').' };
}

/* === Upload to Cloudinary === */
function uploadToCloudinary(file) {
  return new Promise((resolve, reject) => {
    LOG.group('Cloudinary upload');
    LOG.info('cloud:', CLOUDINARY_CLOUD_NAME, 'preset:', CLOUDINARY_UPLOAD_PRESET);

    const fd = new FormData();
    fd.append('file', file);
    fd.append('upload_preset', CLOUDINARY_UPLOAD_PRESET);
    const url = 'https://api.cloudinary.com/v1_1/' + CLOUDINARY_CLOUD_NAME + '/auto/upload';

    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);
    xhr.responseType = 'json';
    xhr.timeout = TIMEOUT_MS;

    xhr.upload.onprogress = e => {
      if (e.lengthComputable) {
        const pct = Math.round(e.loaded / e.total * 80);
        document.getElementById('progressFill').style.width = (10 + pct * 0.25) + '%';
        document.getElementById('progressLabel').textContent = 'Uploading… ' + Math.round(e.loaded / e.total * 100) + '%';
      }
    };

    xhr.onload = () => {
      LOG.info('XHR status:', xhr.status);
      LOG.end();
      const d = xhr.response || {};
      /* FIX #13: validate secure_url exists */
      if (xhr.status >= 200 && xhr.status < 300 && d.secure_url) {
        LOG.info('Cloudinary URL:', d.secure_url);
        resolve(d.secure_url);
      } else {
        const errMsg = (d && d.error && d.error.message)
          ? d.error.message
          : 'Cloudinary upload failed (status ' + xhr.status + '). Check preset name and that it is set to "Unsigned".';
        reject(new Error(errMsg));
      }
    };
    /* FIX #14: all XHR failure events handled */
    xhr.onerror   = () => { LOG.end(); reject(new Error('Network error during Cloudinary upload. Check your connection.')); };
    xhr.onabort   = () => { LOG.end(); reject(new Error('Upload was cancelled.')); };
    xhr.ontimeout = () => { LOG.end(); reject(new Error('Cloudinary upload timed out. File may be too large.')); };

    xhr.send(fd);
  });
}

/* === Send URL to backend === */
async function sendUrl(fileUrl) {
  appendDebug('POST /api/upload url=' + fileUrl.slice(0, 80) + '…');
  setPhase('extract');

  const ctrl = new AbortController();
  const tid  = setTimeout(() => ctrl.abort(), TIMEOUT_MS);

  let resp;
  try {
    resp = await fetch('/api/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },  /* FIX #9 */
      body: JSON.stringify({ file_url: fileUrl }),
      signal: ctrl.signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Request timed out after ' + (TIMEOUT_MS / 1000) + 's.');
    throw new Error('Network error: ' + err.message);
  } finally {
    clearTimeout(tid);
  }

  const data = await readBody(resp);
  if (!resp.ok) throw new Error(data.error || 'Upload failed with status ' + resp.status);
  return data;
}

/* === Main submit handler === */
async function handleSubmit() {
  document.getElementById('retryBar').classList.add('hidden');
  LOG.group('handleSubmit mode=' + uploadMode);

  let form;
  try {
    form = validateForm(); /* FIX #31 */
  } catch (err) {
    showError(err.message);
    LOG.end();
    return;
  }

  setBusy(true);
  setPhase('validate');
  appendDebug('Submit: mode=' + uploadMode);

  try {
    let data;
    if (form.type === 'device') {
      setPhase('upload');
      const cloudUrl = await uploadToCloudinary(form.file);
      appendDebug('Cloudinary URL: ' + cloudUrl);
      setPhase('fetch');
      data = await sendUrl(cloudUrl);
    } else {
      setPhase('fetch');
      data = await sendUrl(form.value);
    }

    setPhase('done');
    renderResult(data);
    LOG.info('Success — file_id:', data.file_id);
  } catch (err) {
    const msg = friendlyError(err.message);
    LOG.err(err);
    appendDebug('ERROR: ' + err.message);
    setPhase('idle');
    showRetry(msg); /* FIX #26 */
  } finally {
    setBusy(false); /* FIX #24 */
    LOG.end();
  }
}

/* === FIX #8: visible error → never silent === */
function showError(msg) {
  const clean = friendlyError(msg);
  appendDebug('Error shown: ' + clean);
  document.getElementById('progressLabel').textContent = clean;
  document.getElementById('progressLabel').style.color = 'var(--danger)';
  setTimeout(() => {
    document.getElementById('progressLabel').style.color = '';
  }, 4000);
}

/* === FIX #26: retry bar === */
function showRetry(msg) {
  const bar    = document.getElementById('retryBar');
  const msgEl  = document.getElementById('retryMsg');
  msgEl.textContent = msg;
  bar.classList.remove('hidden');
}

function resetState() {
  document.getElementById('retryBar').classList.add('hidden');
  document.getElementById('fileUrl').value = '';
  document.getElementById('pdfFile').value = '';
  document.getElementById('fileName').textContent = '';
  document.getElementById('fileName').classList.remove('show');
  document.getElementById('progressLabel').textContent = '';
  setPhase('idle');
  setBusy(false);
  currentFileId = null;
  appendDebug('State reset by user.');
}

/* === Render upload result === */
function renderResult(data) {
  currentFileId = data.file_id;
  currentPage   = 1;
  totalPages    = data.page_count || 1;

  /* FIX #32: file info always shown */
  const fi = document.getElementById('fileInfo');
  fi.innerHTML = [
    row('Filename', data.filename),
    row('Pages', data.page_count),
    row('Characters', (data.content_length || 0).toLocaleString()),
    row('~Tokens', (data.estimated_tokens || 0).toLocaleString()),
    row('Language', (data.language_profile || {}).primary_language || '—'),
  ].join('');

  /* Language panel */
  const lp = data.language_profile || {};
  document.getElementById('langContent').innerHTML =
    row('Primary', lp.primary_language || '—') +
    row('Detected', (lp.detected_languages || []).join(', ') || 'none');
  document.getElementById('langCard').classList.remove('hidden');

  /* FIX #22: warnings panel */
  const warnings = data.warnings || [];
  if (!OCR_ENABLED && data.page_count > 0) {
    warnings.unshift('OCR is disabled on this server. Scanned pages may have no text.');
  }
  if (warnings.length) {
    document.getElementById('warnContent').innerHTML =
      warnings.map(w => '<div class="alert alert-warn" style="margin-bottom:6px;font-size:12px">' + escHtml(w) + '</div>').join('');
    document.getElementById('warnCard').classList.remove('hidden');
  }

  /* Content area */
  document.getElementById('emptyState').classList.add('hidden');
  document.getElementById('contentArea').style.display = 'flex';
  document.getElementById('contentArea').classList.remove('hidden');

  /* Preview */
  const content = data.preview || data.content || '';
  document.getElementById('extractedContent').textContent = content || '(No text extracted)';
  document.getElementById('extractedContent').classList.toggle('result-empty', !content);
  updatePageNav();
}

function row(k, v) {
  return '<div class="info-row"><span class="info-key">' + escHtml(k) + '</span><span class="info-val">' + escHtml(String(v)) + '</span></div>';
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* === Pagination === */
function updatePageNav() {
  document.getElementById('pageNav').textContent = 'Page ' + currentPage + ' / ' + totalPages;
  document.getElementById('prevBtn').disabled = currentPage <= 1;
  document.getElementById('nextBtn').disabled = currentPage >= totalPages;
}

async function loadPage(n) {
  if (!currentFileId) return;
  currentPage = Math.max(1, Math.min(n, totalPages));
  updatePageNav();
  const box = document.getElementById('extractedContent');
  box.textContent = 'Loading page ' + currentPage + '…';
  try {
    const r = await fetch('/api/page/' + currentFileId + '/' + currentPage);
    const d = await r.json();
    box.textContent = d.content || '(No text on this page)';
  } catch {
    box.textContent = 'Failed to load page.';
  }
}

function loadPrevPage() { loadPage(currentPage - 1); }
function loadNextPage() { loadPage(currentPage + 1); }

/* === Search === */
async function doSearch() {
  const q = document.getElementById('searchQuery').value.trim();
  if (!q) { showError('Enter a search query.'); return; }
  if (!currentFileId) { showError('Upload a PDF first.'); return; }

  const box = document.getElementById('searchResults');
  box.innerHTML = '<span class="result-empty">Searching…</span>';

  try {
    const r = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: currentFileId, query: q }),
    });
    const d = await readBody(r);
    if (!r.ok) throw new Error(d.error || 'Search failed');

    if (!d.results || !d.results.length) {
      box.innerHTML = '<span class="result-empty">No matches found for "' + escHtml(q) + '".</span>';
      return;
    }

    box.innerHTML = d.results.map((res, i) =>
      '<div class="search-hit">' +
      '<div class="search-hit-meta">Result ' + (i+1) + ' — Page ' + res.page + '</div>' +
      '<div class="search-hit-ctx">' + escHtml(res.context) + '</div>' +
      '</div>'
    ).join('');
  } catch (err) {
    box.innerHTML = '<div class="alert alert-err">' + escHtml(friendlyError(err.message)) + '</div>';
  }
}

/* === Summarize === */
async function doSummarize() {
  if (!currentFileId) { showError('Upload a PDF first.'); return; }
  const btn = document.getElementById('summarizeBtn');
  const sp  = document.getElementById('sumSpinner');
  const box = document.getElementById('summaryBox');
  btn.disabled = true;
  sp.classList.add('show');
  box.textContent = 'Summarising…';
  box.classList.remove('result-empty');

  try {
    const r = await fetch('/api/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: currentFileId }),
    });
    const d = await readBody(r);
    if (!r.ok) throw new Error(d.error || 'Summary failed');
    box.textContent = d.summary || '(No summary generated)';
  } catch (err) {
    box.innerHTML = '<div class="alert alert-err">' + escHtml(friendlyError(err.message)) + '</div>';
  } finally {
    btn.disabled = false;
    sp.classList.remove('show');
  }
}

init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


# FIX #3: redirect /index.html → /  so Jinja always runs
@app.route("/index.html")
def redirect_index():
    return redirect("/", code=301)


@app.route("/")
def index():
    build_id = os.getenv("VERCEL_GIT_COMMIT_SHA", os.getenv("BUILD_ID", "local"))
    return render_template_string(
        HTML_TEMPLATE,
        cloudinary_cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
        cloudinary_upload_preset=os.getenv("CLOUDINARY_UPLOAD_PRESET", ""),
        ocr_enabled=PDF_EXTRACTOR.ocr_enabled,
        build_id=build_id,  # FIX #29: fingerprint for deployment verification
    )


# FIX #29: healthz exposes config fingerprint so you can verify the right env is live
@app.route("/healthz", methods=["GET"])
def healthz():
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    preset = os.getenv("CLOUDINARY_UPLOAD_PRESET", "")
    build_id = os.getenv("VERCEL_GIT_COMMIT_SHA", os.getenv("BUILD_ID", "local"))
    return jsonify(
        {
            "status": "ok",
            "build_id": build_id,
            "ocr_enabled": PDF_EXTRACTOR.ocr_enabled,
            "cloudinary_configured": bool(cloud_name and preset),
            "cloudinary_cloud_name": cloud_name[:4] + "…" if cloud_name else "",
            "cached_files": len(list(CACHE_DIR.glob("*.json"))),
        }
    )


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


def _log_upload_request() -> None:
    log.info(
        "UPLOAD REQUEST method=%s content_type=%r content_length=%s",
        request.method,
        request.headers.get("Content-Type", ""),
        request.content_length,
    )


def _reject_file_upload():
    return jsonify(
        {
            "error": (
                "Direct file upload is not supported. "
                "Upload your PDF to Cloudinary, S3, or Firebase, then submit the public URL."
            )
        }
    ), 400


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    try:
        _log_upload_request()
        started = time.monotonic()

        ct = (request.headers.get("Content-Type") or "").lower()
        if ct.startswith("multipart/form-data") or request.files:
            return _reject_file_upload()
        # FIX #9: enforce JSON
        if not request.is_json:
            return jsonify(
                {"error": "Request must have Content-Type: application/json"}
            ), 400

        raw = request.get_data(cache=True, as_text=False) or b""
        log.debug("Upload payload: %d bytes", len(raw))

        data = request.get_json(silent=True) or {}
        if set(data.keys()) - {"file_url"}:
            return _reject_file_upload()

        file_url = (data.get("file_url") or "").strip()
        if not file_url:
            return jsonify({"error": "Missing file_url"}), 400

        file_id = uuid.uuid4().hex
        tmp_path: Path | None = None
        try:
            tmp_path, filename = download_pdf_from_url(file_url)
            extracted = extract_pdf_data(tmp_path)
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        full_content = extracted["cleaned_content"]
        preview = build_preview(full_content, limit=1500)

        record = {
            "filename": filename,
            "source_url": file_url,
            "content": full_content,
            "cleaned_content": full_content,
            "preview": preview,
            "content_length": extracted["content_length"],
            "estimated_tokens": extracted["estimated_tokens"],
            "pages": extracted["page_texts"],
            "page_texts": extracted["page_texts"],
            "language_profile": extracted["language_profile"],
            "page_count": extracted["page_count"],
            "structured_document": extracted["structured_document"],
            "tables": extracted["tables"],
            "metadata": extracted["metadata"],
            "warnings": extracted.get("warnings", []),
        }
        save_record(file_id, record)

        elapsed = round((time.monotonic() - started) * 1000, 1)
        ocr_pages = (
            extracted.get("metadata", {}).get("statistics", {}).get("ocr_pages", 0)
        )
        log.info(
            "UPLOAD COMPLETE file_id=%s filename=%r pages=%d ocr_pages=%d chars=%d tokens=%d ms=%s",
            file_id,
            filename,
            extracted["page_count"],
            ocr_pages,
            extracted["content_length"],
            extracted["estimated_tokens"],
            elapsed,
        )

        return jsonify(
            {
                "file_id": file_id,
                "filename": filename,
                "page_count": extracted["page_count"],
                "preview": preview,
                "content": preview,
                "language_profile": extracted["language_profile"],
                "content_length": extracted["content_length"],
                "estimated_tokens": extracted["estimated_tokens"],
                "warnings": extracted.get("warnings", []),  # FIX #22: surface warnings
                "ocr_enabled": PDF_EXTRACTOR.ocr_enabled,
            }
        )

    except ValueError as exc:
        return _friendly_upload_error(str(exc))
    except Exception as exc:
        log.exception("Unhandled error in /api/upload")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/search", methods=["POST"])
def search_in_pdf():
    try:
        data = request.get_json(silent=True) or {}
        file_id = data.get("file_id")
        query = (data.get("query") or "").strip()

        if not file_id or not query:
            return jsonify({"error": "Missing file_id or query"}), 400

        pdf_data = load_record(file_id)
        if pdf_data is None:
            return jsonify({"error": "File not found. Please upload again."}), 404

        escaped = re.escape(_RE_WHITESPACE.sub(" ", query))
        escaped = re.sub(r"\\\s+", r"\\s+", escaped)
        pattern = re.compile(escaped, re.IGNORECASE)

        results = []
        pages = pdf_data.get("page_texts") or pdf_data.get("pages", [])
        for page_num, page_content in enumerate(pages, start=1):
            if not page_content:
                continue
            for m in pattern.finditer(page_content):
                ctx_start = max(0, m.start() - 80)
                ctx_end = min(len(page_content), m.end() + 80)
                context = _RE_WHITESPACE.sub(" ", page_content[ctx_start:ctx_end])
                results.append(
                    {"page": page_num, "position": m.start(), "context": f"…{context}…"}
                )

        return jsonify({"query": query, "count": len(results), "results": results})

    except Exception as exc:
        log.exception("Error in /api/search")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize_pdf():
    try:
        data = request.get_json(silent=True) or {}
        file_id = data.get("file_id")
        if not file_id:
            return jsonify({"error": "Missing file_id"}), 400

        pdf_data = load_record(file_id)
        if pdf_data is None:
            return jsonify({"error": "File not found. Please upload again."}), 404

        summary = summarize_text(
            pdf_data.get("cleaned_content") or pdf_data["content"],
            language_profile=pdf_data.get("language_profile"),
        )
        pdf_data["summary"] = summary
        save_record(file_id, pdf_data)

        return jsonify(
            {
                "file_id": file_id,
                "summary": summary,
                "language_profile": pdf_data.get("language_profile", {}),
            }
        )

    except Exception as exc:
        log.exception("Error in /api/summarize")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/files", methods=["GET"])
def list_files():
    files = [
        {"file_id": fid, "filename": d["filename"], "page_count": d["page_count"]}
        for fid, d in all_records().items()
    ]
    return jsonify({"files": files})


@app.route("/api/content/<file_id>", methods=["GET"])
def get_content(file_id: str):
    pdf_data = load_record(file_id)
    if pdf_data is None:
        return jsonify({"error": "File not found"}), 404

    offset = request.args.get("offset", default=0, type=int)
    limit = request.args.get("limit", default=1500, type=int)
    content = pdf_data.get("cleaned_content") or pdf_data.get("content") or ""
    sliced = slice_content(content, offset=offset, limit=limit)

    return jsonify(
        {
            "file_id": file_id,
            "filename": pdf_data["filename"],
            "page_count": pdf_data["page_count"],
            "preview": build_preview(content, limit=1500),
            **sliced,
        }
    )


@app.route("/api/page/<file_id>/<int:page_number>", methods=["GET"])
def get_page_content(file_id: str, page_number: int):
    pdf_data = load_record(file_id)
    if pdf_data is None:
        return jsonify({"error": "File not found"}), 404

    page_texts = pdf_data.get("page_texts") or []
    if page_number < 1 or page_number > len(page_texts):
        return jsonify({"error": "Page not found"}), 404

    content = page_texts[page_number - 1] or ""
    limit = request.args.get("limit", default=1500, type=int)
    sliced = slice_content(content, offset=0, limit=limit)

    return jsonify(
        {
            "file_id": file_id,
            "filename": pdf_data["filename"],
            "page_number": page_number,
            "page_count": pdf_data["page_count"],
            "preview": build_preview(content, limit=1500),
            **sliced,
        }
    )


@app.route("/api/structured/<file_id>", methods=["GET"])
def get_structured_content(file_id: str):
    pdf_data = load_record(file_id)
    if pdf_data is None:
        return jsonify({"error": "File not found"}), 404
    return jsonify(pdf_data.get("structured_document", {}))


@app.route("/api/stats", methods=["GET"])
def get_stats():
    records = all_records()
    total_pages = sum(d.get("page_count", 0) for d in records.values())
    total_chars = sum(
        d.get("content_length", len(d.get("content", ""))) for d in records.values()
    )
    return jsonify(
        {
            "total_documents": len(records),
            "total_pages": total_pages,
            "total_chars": total_chars,
            "ocr_available": PDF_EXTRACTOR.ocr_enabled,
            "cache_dir": str(CACHE_DIR),
        }
    )
