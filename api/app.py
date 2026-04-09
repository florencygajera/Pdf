"""Flask API for PDF Content Extraction and Search."""

import gzip
import json
import ipaddress
import re
import tempfile
import uuid
import socket
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from pdf import IndianLanguagePDFExtractor


app = Flask(__name__)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_REMOTE_PDF_BYTES = 25 * 1024 * 1024  # 25 MB download cap for Vercel-safe processing
DOWNLOAD_TIMEOUT_SECONDS = 20
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["JSON_SORT_KEYS"] = False

STORAGE_DIR = Path(tempfile.gettempdir()) / "pdf_content_extractor"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CACHE_DIR = STORAGE_DIR / "cache"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for warm invocations.
pdf_storage = {}
PDF_EXTRACTOR = IndianLanguagePDFExtractor()

SCRIPT_RANGES = {
    "hindi": [(0x0900, 0x097F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "english": [(0x0041, 0x007A)],
}

STOPWORDS = {
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


def record_path(file_id: str) -> Path:
    return CACHE_DIR / f"{file_id}.json"


def upload_path(file_id: str, filename: str) -> Path:
    safe_name = secure_filename(filename) or "upload.pdf"
    return UPLOAD_DIR / f"{file_id}_{safe_name}"


def save_record(file_id: str, record: dict) -> None:
    pdf_storage[file_id] = record
    record_path(file_id).write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )


def load_record(file_id: str):
    cached = pdf_storage.get(file_id)
    if cached is not None:
        return cached

    path = record_path(file_id)
    if not path.exists():
        return None

    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    pdf_storage[file_id] = cached
    return cached


def all_records():
    records = {}
    for file_id, data in pdf_storage.items():
        records[file_id] = data

    for path in CACHE_DIR.glob("*.json"):
        file_id = path.stem
        if file_id in records:
            continue
        try:
            records[file_id] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

    return records


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def build_preview(text: str, limit: int = 1500) -> str:
    return compact_text(text)[:limit]


def slice_content(text: str, offset: int = 0, limit: int = 1500) -> dict:
    normalized = compact_text(text)
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 1500), 2000))
    end = min(len(normalized), offset + limit)
    return {
        "content": normalized[offset:end],
        "offset": offset,
        "limit": limit,
        "total_length": len(normalized),
        "has_more": end < len(normalized),
    }


@app.after_request
def compress_json_response(response):
    if response.direct_passthrough:
        return response

    if response.mimetype == "application/json" and "content-encoding" not in response.headers:
        accept_encoding = request.headers.get("Accept-Encoding", "")
        payload = response.get_data()
        if "gzip" in accept_encoding and len(payload) > 1024:
            response.set_data(gzip.compress(payload))
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(response.get_data()))
            response.headers.add("Vary", "Accept-Encoding")

    return response


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
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        if not ip.is_global:
            return False

    return True


def _filename_from_url_or_headers(file_url: str, headers) -> str:
    content_disposition = headers.get("Content-Disposition", "") if headers else ""
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', content_disposition, re.IGNORECASE)
    if match:
        return secure_filename(unquote(match.group(1))) or "upload.pdf"

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

    request_headers = {
        "User-Agent": "PDF-Content-Extractor/1.0",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
    }
    req = Request(file_url, headers=request_headers)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = Path(temp_file.name)
    total_bytes = 0
    filename = "upload.pdf"

    try:
        with urlopen(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            filename = _filename_from_url_or_headers(file_url, response.headers)
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    content_length_value = int(content_length)
                except (TypeError, ValueError):
                    raise ValueError("Remote file size is invalid.") from None
                if content_length_value > MAX_REMOTE_PDF_BYTES:
                    raise ValueError("Remote file is too large.")

            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_REMOTE_PDF_BYTES:
                    raise ValueError("Remote file is too large.")

                temp_file.write(chunk)

        temp_file.flush()
        temp_file.close()

        with temp_path.open("rb") as file_handle:
            header = file_handle.read(8)
            if not header.startswith(b"%PDF-"):
                raise ValueError("Downloaded file is not a valid PDF.")

        return temp_path, filename
    except Exception:
        temp_file.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "").replace("\xa0", " ")).strip()


def clean_text(text: str) -> str:
    return PDF_EXTRACTOR.clean_text(text or "")


def build_search_pattern(query: str):
    escaped = re.escape(normalize_whitespace(query))
    escaped = re.sub(r"\\\s+", r"\\s+", escaped)
    return re.compile(escaped, flags=re.IGNORECASE)


def extract_search_context(text: str, start: int, end: int, window: int = 80) -> str:
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)
    context = normalize_whitespace(text[context_start:context_end])
    return f"...{context}..."


def detect_languages(text: str) -> dict:
    counts = {"english": 0, "hindi": 0, "gujarati": 0}
    for char in text:
        code = ord(char)
        for language, ranges in SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code <= end:
                    counts[language] += 1
                    break

    detected = [language for language, count in counts.items() if count > 0]
    primary = max(counts, key=counts.get) if any(counts.values()) else "unknown"
    if not detected:
        primary = "unknown"

    if len([language for language, count in counts.items() if count > 0]) > 1:
        primary = "mixed"

    return {
        "primary_language": primary,
        "detected_languages": detected,
        "script_counts": counts,
    }


def summarize_text(
    text: str,
    language_profile: dict | None = None,
    max_sentences: int = 3,
    max_chars: int = 700,
) -> str:
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if not cleaned_text:
        return "No readable text was extracted from the PDF."

    sentence_candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned_text)
        if sentence.strip()
    ]

    if len(sentence_candidates) <= max_sentences:
        summary = " ".join(sentence_candidates)
        return summary[:max_chars].strip()

    primary_language = (language_profile or {}).get("primary_language", "unknown")
    if primary_language in {"hindi", "gujarati", "mixed", "unknown"}:
        summary = " ".join(sentence_candidates[:max_sentences])
        return summary[:max_chars].strip()

    words = re.findall(r"\w+", cleaned_text.lower(), flags=re.UNICODE)
    english_words = [word for word in words if word not in STOPWORDS["english"]]
    if not english_words:
        return " ".join(sentence_candidates[:max_sentences])[:max_chars].strip()

    frequency = {}
    for word in english_words:
        frequency[word] = frequency.get(word, 0) + 1

    ranked_sentences = []
    for index, sentence in enumerate(sentence_candidates):
        sentence_words = re.findall(r"\w+", sentence.lower(), flags=re.UNICODE)
        score = sum(frequency.get(word, 0) for word in sentence_words)
        score += min(len(sentence_words), 40) * 0.1
        ranked_sentences.append((score, index, sentence))

    ranked_sentences.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(ranked_sentences[:max_sentences], key=lambda item: item[1])
    summary = " ".join(sentence for _, _, sentence in selected)
    return summary[:max_chars].strip()


def extract_pdf_data(saved_path: Path) -> dict:
    extracted = PDF_EXTRACTOR.extract_from_pdf(str(saved_path), extract_images=True)

    page_texts = []
    raw_page_texts = []
    tables = []
    structured_sections = []
    document_title = normalize_whitespace(extracted.get("metadata", {}).get("title", ""))

    for page in extracted.get("pages", []):
        page_number = page.get("page_number", len(page_texts) + 1)
        cleaned_text = normalize_whitespace(page.get("cleaned_text", ""))
        raw_text = normalize_whitespace(page.get("raw_text", "") or page.get("direct_text", ""))
        page_texts.append(cleaned_text or raw_text)
        raw_page_texts.append(raw_text or cleaned_text)

        page_heading = normalize_whitespace(page.get("page_heading", ""))
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

        for table in page_tables:
            tables.append(
                {
                    "page": page_number,
                    "heading": section_heading,
                    "columns": table.get("columns", []),
                    "rows": table.get("rows", []),
                }
            )

    structured_document = {
        "document_title": document_title or extracted["filename"],
        "sections": structured_sections,
    }

    cleaned_content = "\n\n".join(part for part in page_texts if part.strip()).strip()
    raw_content = "\n\n".join(part for part in raw_page_texts if part.strip()).strip() or cleaned_content
    language_profile = detect_languages(cleaned_content or raw_content)
    metadata = extracted.get("metadata", {})
    metadata["statistics"] = extracted.get("statistics", {})
    metadata["warnings"] = extracted.get("warnings", [])

    return {
        "page_count": extracted.get("metadata", {}).get("page_count", len(extracted.get("pages", []))),
        "full_content": cleaned_content,
        "content": raw_content,
        "cleaned_content": cleaned_content,
        "page_texts": page_texts,
        "language_profile": language_profile,
        "structured_document": structured_document,
        "tables": tables,
        "metadata": metadata,
        "pages_data": extracted.get("pages", []),
    }


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "File too large. Maximum size is 5 MB for this deployment."}), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Not found"}), 404


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PDF Content Extractor & Search</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #f3f4f6;
            --surface: rgba(255, 255, 255, 0.88);
            --surface-strong: #ffffff;
            --text: #111827;
            --muted: #6b7280;
            --line: #e5e7eb;
            --accent: #111827;
            --accent-soft: #374151;
        }

        * {
            box-sizing: border-box;
        }

        html, body {
            height: 100%;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(17, 24, 39, 0.08), transparent 32%),
                linear-gradient(180deg, #fafafa 0%, var(--bg) 100%);
        }

        .container {
            min-height: 100vh;
            width: 100%;
            padding: 28px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .hero {
            background: var(--surface);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(229, 231, 235, 0.9);
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 18px 50px rgba(17, 24, 39, 0.06);
        }

        .hero-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
        }

        h1 {
            margin: 0;
            font-size: clamp(2rem, 3vw, 3.2rem);
            letter-spacing: -0.04em;
            line-height: 1.02;
        }

        .subtitle {
            margin: 10px 0 0;
            max-width: 62ch;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
        }

        .hero-badge {
            align-self: flex-start;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 8px 12px;
            color: var(--accent);
            font-size: 0.9rem;
            background: rgba(255, 255, 255, 0.72);
        }

        .layout {
            width: 100%;
            display: grid;
            grid-template-columns: minmax(320px, 380px) minmax(0, 1fr);
            gap: 20px;
            flex: 1;
        }

        .stack {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .panel {
            background: var(--surface-strong);
            border: 1px solid var(--line);
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 12px 32px rgba(17, 24, 39, 0.04);
        }

        .panel h3 {
            margin: 0 0 14px;
            font-size: 1rem;
            letter-spacing: 0.01em;
        }

        .btn {
            background: var(--accent);
            color: white;
            padding: 12px 16px;
            border: 0;
            border-radius: 12px;
            cursor: pointer;
            font: inherit;
            transition: transform 0.15s ease, background 0.15s ease, opacity 0.15s ease;
        }
        .btn:hover { background: #1f2937; transform: translateY(-1px); }
        .btn:disabled { background: #9ca3af; cursor: not-allowed; transform: none; }

        .btn-secondary {
            background: #e5e7eb;
            color: #111827;
        }
        .btn-secondary:hover { background: #d1d5db; }

        .row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }

        input[type="file"], input[type="text"] {
            padding: 12px 14px;
            font-size: 15px;
            border-radius: 12px;
            border: 1px solid var(--line);
            background: white;
            width: 100%;
            font: inherit;
            color: var(--text);
        }

        input[type="text"] { flex: 1; min-width: 220px; }

        .result {
            white-space: pre-wrap;
            background: #fafafa;
            border: 1px solid var(--line);
            padding: 16px;
            border-radius: 14px;
            margin-top: 14px;
            min-height: 160px;
            max-height: 62vh;
            overflow: auto;
            line-height: 1.6;
        }

        .hidden { display: none; }
        .muted { color: var(--muted); }

        .full-height {
            min-height: 100%;
        }

        .content-panel {
            display: flex;
            flex-direction: column;
            min-height: 0;
        }

        .content-panel .result {
            flex: 1;
        }

        .file-info {
            font-size: 0.95rem;
        }

        @media (max-width: 980px) {
            .layout {
                grid-template-columns: 1fr;
            }

            .hero {
                padding: 22px;
            }
        }

        @media (max-width: 640px) {
            .container {
                padding: 14px;
            }

            .hero, .panel {
                border-radius: 18px;
                padding: 18px;
            }

            .row {
                flex-direction: column;
                align-items: stretch;
            }

            .btn {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <section class="hero">
            <div class="hero-top">
                <div>
                    <h1>PDF Content Extractor</h1>
                    <p class="subtitle">Upload a PDF, extract readable text, inspect the detected language, summarize the document, and search inside it from one clean full-page workspace.</p>
                </div>
                <div class="hero-badge">Minimal full-page view</div>
            </div>
        </section>

        <div class="layout">
            <div class="stack">
                <div class="panel">
                    <h3>Upload PDF</h3>
                    <form id="uploadForm">
                        <div class="row">
                            <input type="text" id="fileUrl" name="file_url" placeholder="Paste Cloudinary, S3, or Firebase PDF URL" required>
                            <button type="submit" class="btn">Fetch & Extract</button>
                        </div>
                    </form>
                    <div class="muted" style="margin-top: 8px;">Upload the PDF to cloud storage first, then paste its public URL here.</div>
                    <div id="uploadStatus" class="muted" style="margin-top: 10px;"></div>
                </div>

                <div class="panel">
                    <h3>File Info</h3>
                    <div id="fileInfo" class="muted file-info">No file uploaded yet.</div>
                </div>

                <div id="languagePanel" class="panel hidden">
                    <h3>Detected Language</h3>
                    <div id="languageContent" class="result"></div>
                </div>

                <div id="summaryActions" class="panel hidden">
                    <h3>Summary</h3>
                    <div class="row">
                        <button class="btn" id="summarizeButton" onclick="summarizePDF()">Summarise Whole PDF</button>
                    </div>
                    <div id="summaryStatus" class="muted" style="margin-top: 10px;"></div>
                    <div id="summaryContent" class="result"></div>
                </div>
            </div>

            <div class="stack full-height">
                <div id="resultPanel" class="panel hidden content-panel">
                    <h3>Extracted Content</h3>
                    <div id="extractedContent" class="result"></div>
                </div>

                <div id="searchPanel" class="panel hidden content-panel">
                    <h3>Search in PDF</h3>
                    <div class="row">
                        <input type="text" id="searchQuery" placeholder="Enter text to search...">
                        <button class="btn btn-secondary" onclick="searchInPDF()">Search</button>
                    </div>
                    <div id="searchResults" class="result"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFileId = null;

        async function readResponseBody(response) {
            const contentType = response.headers.get('content-type') || '';

            if (contentType.includes('application/json')) {
                try {
                    return await response.json();
                } catch (error) {
                    return { error: 'The server returned invalid JSON.' };
                }
            }

            const text = await response.text();
            return {
                error: text.trim() || `Request failed with status ${response.status}.`
            };
        }

        document.getElementById('uploadForm').addEventListener('submit', async (event) => {
            event.preventDefault();

            const fileUrlInput = document.getElementById('fileUrl');
            const fileUrl = fileUrlInput.value.trim();

            if (!fileUrl) {
                alert('Please provide a cloud PDF URL');
                return;
            }

            document.getElementById('uploadStatus').textContent = 'Processing...';

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_url: fileUrl })
                });

                const data = await readResponseBody(response);

                if (!response.ok) {
                    throw new Error(data.error || 'Upload failed');
                }

                currentFileId = data.file_id;
                document.getElementById('uploadStatus').textContent = 'Document loaded successfully';
                document.getElementById('fileInfo').textContent = `File: ${data.filename} (${data.page_count} pages)`;
                document.getElementById('languageContent').textContent = JSON.stringify(data.language_profile || {}, null, 2);
                document.getElementById('languagePanel').classList.remove('hidden');
                document.getElementById('summaryActions').classList.remove('hidden');
                document.getElementById('summaryStatus').textContent = 'Click "Summarise Whole PDF" to generate a summary from the full extracted text.';
                document.getElementById('summaryContent').textContent = '';
                document.getElementById('extractedContent').textContent = data.preview || data.content || '';
                document.getElementById('resultPanel').classList.remove('hidden');
                document.getElementById('searchPanel').classList.remove('hidden');
            } catch (error) {
                document.getElementById('uploadStatus').textContent = `Error: ${error.message}`;
            }
        });

        async function searchInPDF() {
            const query = document.getElementById('searchQuery').value.trim();

            if (!query) {
                alert('Please enter a search query');
                return;
            }

            if (!currentFileId) {
                alert('Please upload a PDF first');
                return;
            }

            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_id: currentFileId, query })
                });

                const data = await readResponseBody(response);

                if (!response.ok) {
                    throw new Error(data.error || 'Search failed');
                }

                if (!data.results.length) {
                    document.getElementById('searchResults').textContent = 'No matches found.';
                    return;
                }

                let output = `Found ${data.results.length} result(s):\n\n`;
                data.results.forEach((result, index) => {
                    output += `Result ${index + 1}\n`;
                    output += `Page: ${result.page}\n`;
                    output += `Context: ${result.context}\n\n`;
                });
                document.getElementById('searchResults').textContent = output;
            } catch (error) {
                document.getElementById('searchResults').textContent = `Error: ${error.message}`;
            }
        }

        async function summarizePDF() {
            if (!currentFileId) {
                alert('Please upload a PDF first');
                return;
            }

            const button = document.getElementById('summarizeButton');
            const status = document.getElementById('summaryStatus');
            const summaryBox = document.getElementById('summaryContent');

            button.disabled = true;
            status.textContent = 'Summarising full document...';
            summaryBox.textContent = '';

            try {
                const response = await fetch('/api/summarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_id: currentFileId })
                });

                const data = await readResponseBody(response);

                if (!response.ok) {
                    throw new Error(data.error || 'Summary failed');
                }

                summaryBox.textContent = data.summary || 'No summary available.';
                status.textContent = 'Summary generated from the full PDF content.';
            } catch (error) {
                status.textContent = `Error: ${error.message}`;
            } finally {
                button.disabled = false;
            }
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    try:
        data = request.get_json(silent=True) or {}
        file_url = (data.get("file_url") or "").strip()
        if not file_url:
            return jsonify({"error": "Missing file_url"}), 400

        file_id = uuid.uuid4().hex
        downloaded_path = None
        try:
            downloaded_path, filename = download_pdf_from_url(file_url)
            extracted = extract_pdf_data(downloaded_path)
        finally:
            if downloaded_path and downloaded_path.exists():
                downloaded_path.unlink(missing_ok=True)

        full_content = extracted["cleaned_content"]
        preview = build_preview(full_content, limit=1500)
        record = {
            "filename": filename,
            "source_url": file_url,
            "content": full_content,
            "cleaned_content": full_content,
            "preview": preview,
            "pages": extracted["page_texts"],
            "page_texts": extracted["page_texts"],
            "language_profile": extracted["language_profile"],
            "page_count": extracted["page_count"],
            "structured_document": extracted["structured_document"],
            "tables": extracted["tables"],
            "metadata": extracted["metadata"],
        }
        save_record(file_id, record)

        return jsonify(
            {
                "file_id": file_id,
                "filename": filename,
                "page_count": extracted["page_count"],
                "preview": preview,
                "content": preview,
                "language_profile": extracted["language_profile"],
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
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

        pattern = build_search_pattern(query)
        results = []
        page_texts = pdf_data.get("page_texts") or pdf_data.get("pages", [])
        for page_num, page_content in enumerate(page_texts, start=1):
            if not page_content:
                continue
            for match in pattern.finditer(page_content):
                results.append(
                    {
                        "page": page_num,
                        "position": match.start(),
                        "context": extract_search_context(page_content, match.start(), match.end()),
                    }
                )

        return jsonify({"query": query, "count": len(results), "results": results})
    except Exception as exc:
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
        return jsonify({"error": str(exc)}), 500


@app.route("/api/files", methods=["GET"])
def list_files():
    files = []
    for file_id, data in all_records().items():
        files.append(
            {
                "file_id": file_id,
                "filename": data["filename"],
                "page_count": data["page_count"],
            }
        )
    return jsonify({"files": files})


@app.route("/api/content/<file_id>", methods=["GET"])
def get_content(file_id):
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
def get_page_content(file_id, page_number):
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
def get_structured_content(file_id):
    pdf_data = load_record(file_id)
    if pdf_data is None:
        return jsonify({"error": "File not found"}), 404

    return jsonify(pdf_data.get("structured_document", {}))
