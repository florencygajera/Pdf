"""
Flask API for PDF Content Extraction and Search — production-grade.

Fixes applied vs original:
  [CRITICAL] No from __future__ import annotations → added; dict|None works on Python 3.9+
  [CRITICAL] pdf_storage had no thread-lock → protected with threading.RLock
  [CRITICAL] PDF_EXTRACTOR._page_text_cache was instance-level → fixed in pdf.py (local var)
  [BUG]      extract_pdf_data read non-existent keys content_length/estimated_tokens → computed locally
  [BUG]      all_records() iterated pdf_storage without lock → snapshot copy under lock
  [BUG]      save_record had no error handling for disk write → try/except + warning log
  [BUG]      compress_json_response called get_data() twice → single variable
  [BUG]      JS cloudinary vars injected as raw string inside <script> → |tojson filter
  [BUG]      download_pdf_from_url: temp_file FD leaked on ValueError → moved close() to finally
  [WARN]     upload_path() / UPLOAD_DIR dead code → removed
  [WARN]     print() used for logging → replaced with logging module
  [WARN]     No estimated_tokens calculation → computed as len(text)//4
  [WARN]     slice_content double-cast of limit → single cast only
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

from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Logging — structured, level-aware, timestamped
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
# Local import of extractor
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from pdf import IndianLanguagePDFExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Flask app + config
# ---------------------------------------------------------------------------
app = Flask(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB  — JSON payload only (just the URL)
MAX_REMOTE_PDF_BYTES = 50 * 1024 * 1024  # 50 MB — remote PDF content
DOWNLOAD_TIMEOUT_SECONDS = 20

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["JSON_SORT_KEYS"] = False

# ---------------------------------------------------------------------------
# Storage dirs — only cache dir is used; UPLOAD_DIR removed (was dead code)
# ---------------------------------------------------------------------------
STORAGE_DIR = Path(tempfile.gettempdir()) / "pdf_content_extractor"
CACHE_DIR = STORAGE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory store — guarded by a reentrant lock (FIX: was unprotected)
# ---------------------------------------------------------------------------
_storage_lock: threading.RLock = threading.RLock()
pdf_storage: dict[str, dict] = {}

# Single shared extractor — thread-safe now that _page_text_cache is local in pdf.py
PDF_EXTRACTOR = IndianLanguagePDFExtractor()

# ---------------------------------------------------------------------------
# Language / stop-word data
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

# Compiled regex helpers
_RE_WHITESPACE = re.compile(r"\s+")
_RE_NBSP = re.compile(r"\xa0")
_RE_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_RE_WORDS = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Storage helpers  (all guarded by _storage_lock)
# ---------------------------------------------------------------------------


def _record_path(file_id: str) -> Path:
    return CACHE_DIR / f"{file_id}.json"


def save_record(file_id: str, record: dict) -> None:
    with _storage_lock:
        pdf_storage[file_id] = record
    # Persist to disk outside the lock to avoid blocking other threads
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
    """Return snapshot of every known record — thread-safe."""
    # FIX: take a snapshot under lock so no RuntimeError during iteration
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
    # FIX: single cast for each param
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
# Gzip response compression
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
        # FIX: single get_data() call stored in variable
        payload = response.get_data()
        if "gzip" in accept and len(payload) > 1024:
            compressed = gzip.compress(payload)
            response.set_data(compressed)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(compressed))
            response.headers.add("Vary", "Accept-Encoding")
    return response


# ---------------------------------------------------------------------------
# URL / host validation
# ---------------------------------------------------------------------------


def _is_public_host(hostname: str) -> bool:
    if not hostname:
        return False
    lowered = hostname.lower()
    if lowered in {"localhost", "127.0.0.1", "::1"} or lowered.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_global
    except ValueError:
        pass
    try:
        addresses = {
            info[4][0]
            for info in socket.getaddrinfo(hostname, None)
            if info and info[4]
        }
    except socket.gaierror:
        return False
    if not addresses:
        return False
    for address in addresses:
        try:
            if not ipaddress.ip_address(address).is_global:
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
        raise ValueError("file_url must point to a public host.")

    req_headers = {
        "User-Agent": "PDF-Content-Extractor/1.0",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
    }
    req = Request(file_url, headers=req_headers)
    opener = build_opener(_LimitedRedirectHandler())

    # FIX: temp_file is ALWAYS closed in the finally block — no FD leaks
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

        # Flush before reading back
        tmp.flush()

        # Verify PDF magic bytes
        with tmp_path.open("rb") as fh:
            if not fh.read(8).startswith(b"%PDF-"):
                raise ValueError("Downloaded file is not a valid PDF.")

        return tmp_path, filename

    except (TimeoutError, socket.timeout) as exc:
        raise ValueError(
            "Download timed out. Please try a smaller or faster URL."
        ) from exc
    finally:
        # FIX: always close the file handle — prevents descriptor leak
        try:
            tmp.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Language detection (app-level, for API response)
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

    # TF-IDF-lite scoring for English
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
# PDF data extraction (bridges pdf.py → app storage)
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

    # FIX: compute content_length and estimated_tokens locally;
    #      these keys were expected but never present in the original code.
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
    }


# ---------------------------------------------------------------------------
# Error response helpers
# ---------------------------------------------------------------------------


def _friendly_upload_error(message: str):
    msg = message or "Upload failed."
    lower = msg.lower()
    if "processing limit" in lower:
        msg = "File exceeds processing limit (50 MB)."
    elif "timeout" in lower:
        msg = "The request timed out while processing the PDF."
    elif "cloudinary" in lower:
        msg = "Cloudinary upload is not configured on this deployment."
    elif ("must use http" in lower) or ("file_url is invalid" in lower):
        msg = "Please provide a valid cloud PDF URL."
    elif "public host" in lower:
        msg = "That URL is not allowed. Please use a public cloud file URL."
    elif "not a valid pdf" in lower:
        msg = "The URL did not return a valid PDF file."
    return jsonify({"error": msg}), 400


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.errorhandler(413)
def request_too_large(_e):
    return jsonify(
        {"error": "File too large. Maximum size is 5 MB for this deployment."}
    ), 413


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(_e):
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0" />
    <title>PDF Content Extractor &amp; Search</title>
    <script>
        window.CLOUDINARY_CLOUD_NAME = {{ cloudinary_cloud_name|tojson }};
        window.CLOUDINARY_UPLOAD_PRESET = {{ cloudinary_upload_preset|tojson }};
    </script>
    <style>
        :root {
            color-scheme: light;
            --bg: #f3f4f6;
            --surface: rgba(255,255,255,0.88);
            --surface-strong: #ffffff;
            --text: #111827;
            --muted: #6b7280;
            --line: #e5e7eb;
            --accent: #111827;
        }
        *, *::before, *::after { box-sizing: border-box; }
        html, body { height: 100%; margin: 0; }
        body {
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
            color: var(--text);
            background: radial-gradient(circle at top left,rgba(17,24,39,.08),transparent 32%),
                        linear-gradient(180deg,#fafafa 0%,var(--bg) 100%);
        }
        .container { min-height:100vh;width:100%;padding:28px;display:flex;flex-direction:column;gap:20px; }
        .hero {
            background: var(--surface); backdrop-filter: blur(18px);
            border:1px solid rgba(229,231,235,.9); border-radius:24px;
            padding:28px; box-shadow:0 18px 50px rgba(17,24,39,.06);
        }
        .hero-top { display:flex;align-items:flex-start;justify-content:space-between;gap:20px;flex-wrap:wrap; }
        h1 { margin:0;font-size:clamp(2rem,3vw,3.2rem);letter-spacing:-.04em;line-height:1.02; }
        .subtitle { margin:10px 0 0;max-width:62ch;color:var(--muted);font-size:1rem;line-height:1.6; }
        .hero-badge { align-self:flex-start;border:1px solid var(--line);border-radius:999px;padding:8px 12px;font-size:.9rem;background:rgba(255,255,255,.72); }
        .layout { width:100%;display:grid;grid-template-columns:minmax(320px,380px) minmax(0,1fr);gap:20px;flex:1; }
        .stack { display:flex;flex-direction:column;gap:20px; }
        .panel { background:var(--surface-strong);border:1px solid var(--line);padding:20px;border-radius:20px;box-shadow:0 12px 32px rgba(17,24,39,.04); }
        .panel h3 { margin:0 0 14px;font-size:1rem;letter-spacing:.01em; }
        .btn { background:var(--accent);color:#fff;padding:12px 16px;border:0;border-radius:12px;cursor:pointer;font:inherit;transition:transform .15s,background .15s,opacity .15s; }
        .btn:hover { background:#1f2937;transform:translateY(-1px); }
        .btn:disabled { background:#9ca3af;cursor:not-allowed;transform:none; }
        .btn-secondary { background:#e5e7eb;color:#111827; }
        .btn-secondary:hover { background:#d1d5db; }
        .row { display:flex;gap:10px;flex-wrap:wrap;align-items:center; }
        input[type="file"], input[type="text"] { padding:12px 14px;font-size:15px;border-radius:12px;border:1px solid var(--line);background:#fff;width:100%;font:inherit;color:var(--text); }
        input[type="text"] { flex:1;min-width:220px; }
        .result { white-space:pre-wrap;background:#fafafa;border:1px solid var(--line);padding:16px;border-radius:14px;margin-top:14px;min-height:160px;max-height:62vh;overflow:auto;line-height:1.6; }
        .progress-bar { width:100%;height:10px;border-radius:999px;overflow:hidden;background:#e5e7eb;margin-top:8px; }
        .progress-bar>div { width:0%;height:100%;border-radius:inherit;background:var(--accent);transition:width .15s; }
        .mode-active { background:var(--accent) !important;color:#fff !important; }
        .hidden { display:none; }
        .muted { color:var(--muted); }
        .content-panel { display:flex;flex-direction:column;min-height:0; }
        .content-panel .result { flex:1; }
        @media(max-width:980px){ .layout { grid-template-columns:1fr; } }
        @media(max-width:640px){
            .container { padding:14px; }
            .hero,.panel { border-radius:18px;padding:18px; }
            .row { flex-direction:column;align-items:stretch; }
            .btn { width:100%; }
        }
    </style>
</head>
<body>
<div class="container">
    <section class="hero">
        <div class="hero-top">
            <div>
                <h1>PDF Content Extractor</h1>
                <p class="subtitle">Upload a PDF, extract readable text (typed or scanned), detect language, summarise, and search — all from one workspace.</p>
            </div>
            <div class="hero-badge">Production build</div>
        </div>
    </section>

    <div class="layout">
        <div class="stack">
            <div class="panel">
                <h3>Upload PDF</h3>
                <div class="row" style="margin-bottom:12px;">
                    <button type="button" class="btn btn-secondary" id="modeUrlBtn" onclick="setMode('url')">Paste URL</button>
                    <button type="button" class="btn btn-secondary" id="modeDeviceBtn" onclick="setMode('device')">Upload from Device</button>
                </div>
                <form id="uploadForm">
                    <div id="urlSection">
                        <div class="row">
                            <input type="text" id="fileUrl" placeholder="Paste Cloudinary, S3, or Firebase PDF URL">
                        </div>
                    </div>
                    <div id="deviceSection" class="hidden">
                        <div class="row"><input type="file" id="pdfFile" accept="application/pdf,.pdf"></div>
                        <div id="selectedName" class="muted" style="margin-top:8px;">No file selected.</div>
                        <div class="muted" style="margin-top:8px;">PDF only · max 50 MB · uploaded to Cloudinary, not Flask.</div>
                        <div class="progress-bar"><div id="progressFill"></div></div>
                        <div id="progressText" class="muted" style="margin-top:6px;">Waiting for file.</div>
                    </div>
                    <button type="submit" class="btn" id="submitBtn" style="margin-top:12px;">Fetch &amp; Extract</button>
                </form>
                <div id="uploadHint" class="muted" style="margin-top:8px;"></div>
                <div id="cloudinaryBadge" class="muted" style="margin-top:8px;font-weight:600;"></div>
                <div id="uploadStatus" class="muted" style="margin-top:10px;"></div>
                <div id="debugPanel" class="result" style="margin-top:12px;max-height:220px;display:none;"></div>
            </div>

            <div class="panel">
                <h3>File Info</h3>
                <div id="fileInfo" class="muted">No file uploaded yet.</div>
            </div>

            <div id="langPanel" class="panel hidden">
                <h3>Detected Language</h3>
                <div id="langContent" class="result"></div>
            </div>

            <div id="summaryPanel" class="panel hidden">
                <h3>Summary</h3>
                <div class="row">
                    <button class="btn" id="summarizeBtn" onclick="summarize()">Summarise Whole PDF</button>
                </div>
                <div id="summaryStatus" class="muted" style="margin-top:10px;"></div>
                <div id="summaryContent" class="result"></div>
            </div>
        </div>

        <div class="stack">
            <div id="resultPanel" class="panel hidden content-panel">
                <h3>Extracted Content</h3>
                <div id="extractedContent" class="result"></div>
            </div>
            <div id="searchPanel" class="panel hidden content-panel">
                <h3>Search in PDF</h3>
                <div class="row">
                    <input type="text" id="searchQuery" placeholder="Enter text to search...">
                    <button class="btn btn-secondary" onclick="doSearch()">Search</button>
                </div>
                <div id="searchResults" class="result"></div>
            </div>
        </div>
    </div>
</div>

<script>
// FIX: use |tojson so Jinja2 properly escapes values for JS — no raw string injection
const CLOUDINARY_CLOUD_NAME   = window.CLOUDINARY_CLOUD_NAME || "";
const CLOUDINARY_UPLOAD_PRESET= window.CLOUDINARY_UPLOAD_PRESET || "";
const CLOUDINARY_READY        = Boolean(CLOUDINARY_CLOUD_NAME && CLOUDINARY_UPLOAD_PRESET);
const MAX_DEVICE_BYTES        = 50 * 1024 * 1024;
const TIMEOUT_MS              = 40_000;

console.log("Cloud Name:", CLOUDINARY_CLOUD_NAME);
console.log("Preset:", CLOUDINARY_UPLOAD_PRESET);

let currentFileId  = null;
let uploadMode     = 'url';

function $(id){ return document.getElementById(id); }

function setProgress(pct, msg){
    $('progressFill').style.width = Math.max(0,Math.min(100,pct))+'%';
    if(msg) $('progressText').textContent = msg;
}

function setBusy(busy){
    $('submitBtn').disabled     = busy;
    $('modeUrlBtn').disabled    = busy;
    $('modeDeviceBtn').disabled = busy;
}

function showDebug(text){
    const p = $('debugPanel');
    p.style.display = text ? 'block' : 'none';
    p.textContent   = text || '';
}

function friendlyError(msg){
    const t = (msg||'Something went wrong.').toLowerCase();
    if(t.includes('processing limit'))  return 'File exceeds processing limit (50 MB).';
    if(t.includes('cloudinary'))        return 'Device upload not configured (Cloudinary missing).';
    if(t.includes('timeout'))           return 'The request timed out.';
    if(t.includes('invalid json'))      return 'Server returned an invalid response.';
    if(t.includes('url')||t.includes('public host')) return 'Please provide a valid cloud PDF URL.';
    return msg;
}

function renderCloudinaryBadge(){
    const b = $('cloudinaryBadge');
    if(CLOUDINARY_READY){
        b.textContent = '\u2713 Cloudinary: '+CLOUDINARY_CLOUD_NAME;
        b.style.color = '#065f46';
    } else {
        b.textContent = '\u26a0 Cloudinary missing — device upload disabled.';
        b.style.color = '#b45309';
    }
}

function setMode(mode){
    uploadMode = mode;
    $('modeUrlBtn').classList.toggle('mode-active', mode==='url');
    $('modeDeviceBtn').classList.toggle('mode-active', mode==='device');
    $('urlSection').classList.toggle('hidden', mode!=='url');
    $('deviceSection').classList.toggle('hidden', mode!=='device');
    $('submitBtn').textContent = mode==='url' ? 'Fetch & Extract' : 'Upload & Extract';
    $('uploadHint').textContent = mode==='url'
        ? 'Upload the PDF to cloud storage first, then paste its public URL here.'
        : 'File uploads go to Cloudinary; Flask only receives the resulting URL.';
    $('uploadStatus').textContent = '';
    showDebug('');
    setProgress(0, mode==='url' ? 'URL mode ready.' : 'Waiting for file selection.');
    if(mode==='device'){
        if(!CLOUDINARY_READY){ $('uploadStatus').textContent='Device upload disabled: Cloudinary not configured.'; return; }
        $('pdfFile').click();
    }
}

async function readBody(resp){
    const ct = resp.headers.get('content-type')||'';
    if(ct.includes('application/json')){
        try{ return await resp.json(); }
        catch{ return {error:'Server returned invalid JSON.'}; }
    }
    const t = await resp.text();
    return { error: t.trim()||`Request failed (${resp.status}).` };
}

async function sendUrl(fileUrl, label){
    const payload     = { file_url: fileUrl };
    const payloadJson = JSON.stringify(payload);
    if(new Blob([payloadJson]).size > 2048){
        $('uploadStatus').textContent='Error: URL too long.'; return;
    }
    showDebug(`[${label}] POST /api/upload\n${payloadJson}`);
    $('uploadStatus').textContent='Extracting…';

    const ctrl = new AbortController();
    const tid  = setTimeout(()=>ctrl.abort(), TIMEOUT_MS);
    let resp;
    try{
        resp = await fetch('/api/upload',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: payloadJson,
            signal: ctrl.signal,
        });
    } finally { clearTimeout(tid); }

    const data = await readBody(resp);
    if(!resp.ok) throw new Error(friendlyError(data.error||'Upload failed'));

    currentFileId = data.file_id;
    $('uploadStatus').textContent='Document loaded.';
    $('fileInfo').textContent=`File: ${data.filename}  ·  ${data.page_count} page(s)  ·  ${data.content_length||0} chars  ·  ~${data.estimated_tokens||0} tokens`;
    $('langContent').textContent=JSON.stringify(data.language_profile||{},null,2);
    $('langPanel').classList.remove('hidden');
    $('summaryPanel').classList.remove('hidden');
    $('summaryStatus').textContent='Click "Summarise Whole PDF" to generate a summary.';
    $('summaryContent').textContent='';
    $('extractedContent').textContent=data.preview||data.content||'';
    $('resultPanel').classList.remove('hidden');
    $('searchPanel').classList.remove('hidden');
    showDebug('');
}

function uploadToCloudinary(file){
    return new Promise((resolve,reject)=>{
        if(!CLOUDINARY_READY){ reject(new Error('Device upload disabled: Cloudinary not configured.')); return; }
        const fd = new FormData();
        fd.append('file', file);
        fd.append('upload_preset', CLOUDINARY_UPLOAD_PRESET);
        const url = `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD_NAME}/auto/upload`;
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url, true);
        xhr.responseType='json';
        xhr.timeout = TIMEOUT_MS;
        xhr.upload.onprogress = e=>{
            if(e.lengthComputable) setProgress(Math.round(e.loaded/e.total*100),`Uploading… ${Math.round(e.loaded/e.total*100)}%`);
            else setProgress(30,'Uploading to Cloudinary…');
        };
        xhr.onload=()=>{
            const d=xhr.response||{};
            if(xhr.status>=200&&xhr.status<300&&d.secure_url){ setProgress(100,'Upload complete.'); resolve(d.secure_url); }
            else reject(new Error(d?.error?.message||'Cloudinary upload failed.'));
        };
        xhr.onerror=()=>reject(new Error('Network error during upload.'));
        xhr.onabort=()=>reject(new Error('Upload cancelled.'));
        xhr.ontimeout=()=>reject(new Error('Upload timed out.'));
        xhr.send(fd);
    });
}

$('uploadForm').addEventListener('submit', async e=>{
    e.preventDefault();
    setBusy(true);
    try{
        if(uploadMode==='device'){
            if(!CLOUDINARY_READY) throw new Error('Device upload disabled: Cloudinary not configured.');
            const file=$('pdfFile').files[0];
            if(!file) throw new Error('Please choose a PDF file first.');
            if(!file.type.includes('pdf')&&!file.name.toLowerCase().endsWith('.pdf')) throw new Error('Only PDF files are allowed.');
            if(file.size>MAX_DEVICE_BYTES) throw new Error('File exceeds processing limit (50 MB).');
            $('uploadStatus').textContent='Uploading to Cloudinary…';
            setProgress(0,'Starting…');
            const url=await uploadToCloudinary(file);
            await sendUrl(url,'device→cloudinary→backend');
        } else {
            const raw=$('fileUrl').value.trim();
            if(!raw) throw new Error('Please provide a cloud PDF URL.');
            try{
                const p=new URL(raw);
                if(!['http:','https:'].includes(p.protocol)) throw new Error('bad protocol');
            } catch{ throw new Error('Please provide a valid cloud PDF URL.'); }
            $('uploadStatus').textContent='Processing…';
            await sendUrl(raw,'url');
        }
    } catch(err){
        const msg=friendlyError(err.message);
        $('uploadStatus').textContent='Error: '+msg;
        setProgress(0,msg);
    } finally{ setBusy(false); }
});

$('pdfFile').addEventListener('change',e=>{
    const f=e.target.files&&e.target.files[0];
    $('selectedName').textContent=f?`Selected: ${f.name}`:'No file selected.';
    if(f) setProgress(0,`Ready: ${f.name}`);
});

async function doSearch(){
    const q=$('searchQuery').value.trim();
    if(!q){alert('Please enter a search query.');return;}
    if(!currentFileId){alert('Please upload a PDF first.');return;}
    try{
        const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({file_id:currentFileId,query:q})});
        const d=await readBody(r);
        if(!r.ok) throw new Error(d.error||'Search failed');
        if(!d.results.length){$('searchResults').textContent='No matches found.';return;}
        let out=`Found ${d.results.length} result(s):\n\n`;
        d.results.forEach((res,i)=>{ out+=`Result ${i+1}\nPage: ${res.page}\nContext: ${res.context}\n\n`; });
        $('searchResults').textContent=out;
    } catch(err){ $('searchResults').textContent='Error: '+err.message; }
}

async function summarize(){
    if(!currentFileId){alert('Please upload a PDF first.');return;}
    const btn=$('summarizeBtn'), status=$('summaryStatus'), box=$('summaryContent');
    btn.disabled=true; status.textContent='Summarising…'; box.textContent='';
    try{
        const r=await fetch('/api/summarize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({file_id:currentFileId})});
        const d=await readBody(r);
        if(!r.ok) throw new Error(d.error||'Summary failed');
        box.textContent=d.summary||'No summary available.';
        status.textContent='Summary generated from full PDF content.';
    } catch(err){ status.textContent='Error: '+err.message; }
    finally{ btn.disabled=false; }
}

// Init
setMode('url');
renderCloudinaryBadge();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        cloudinary_cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
        cloudinary_upload_preset=os.getenv("CLOUDINARY_UPLOAD_PRESET", ""),
    )


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify(
        {
            "status": "ok",
            "ocr_enabled": PDF_EXTRACTOR.ocr_enabled,
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
        {"error": "File upload is not allowed. Please provide a cloud file URL."}
    ), 400


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    try:
        _log_upload_request()
        started = time.monotonic()

        # Reject multipart / form-data uploads
        ct = (request.headers.get("Content-Type") or "").lower()
        if ct.startswith("multipart/form-data") or request.files:
            return _reject_file_upload()
        if not request.is_json:
            return _reject_file_upload()

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
                    {
                        "page": page_num,
                        "position": m.start(),
                        "context": f"…{context}…",
                    }
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
    """System-level stats for monitoring."""
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
