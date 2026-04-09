"""Flask API for PDF Content Extraction and Search."""

import json
import re
import tempfile
import uuid
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from pdf import IndianLanguagePDFExtractor


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
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
    content_parts = []
    tables = []
    structured_sections = []
    document_title = normalize_whitespace(extracted.get("metadata", {}).get("title", ""))

    for page in extracted.get("pages", []):
        page_number = page.get("page_number", len(page_texts) + 1)
        cleaned_text = normalize_whitespace(page.get("cleaned_text", ""))
        raw_text = normalize_whitespace(page.get("raw_text", "") or page.get("direct_text", ""))
        page_texts.append(cleaned_text or raw_text)
        raw_page_texts.append(raw_text or cleaned_text)
        content_parts.append(f"--- Page {page_number} ---\n{cleaned_text or raw_text}")

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
        "content_parts": content_parts,
        "structured_document": structured_document,
        "tables": tables,
        "metadata": metadata,
        "pages_data": extracted.get("pages", []),
    }


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


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
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 28px;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        }
        h1 { margin-top: 0; }
        .panel {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .btn {
            background: #2563eb;
            color: white;
            padding: 10px 16px;
            border: 0;
            border-radius: 8px;
            cursor: pointer;
        }
        .btn:hover { background: #1d4ed8; }
        .btn:disabled { background: #9ca3af; cursor: not-allowed; }
        .row { display: flex; gap: 10px; flex-wrap: wrap; }
        input[type="file"], input[type="text"] {
            padding: 10px;
            font-size: 15px;
        }
        input[type="text"] { flex: 1; min-width: 220px; }
        .result {
            white-space: pre-wrap;
            background: #fafafa;
            border: 1px solid #e5e7eb;
            padding: 14px;
            border-radius: 8px;
            margin-top: 14px;
            max-height: 420px;
            overflow: auto;
        }
        .hidden { display: none; }
        .muted { color: #6b7280; }
    </style>
</head>
<body>
    <div class="container">
        <h1>PDF Content Extractor & Search</h1>
        <p class="muted">Upload a PDF, extract text, and search within the uploaded document.</p>

        <div class="panel">
            <h3>Upload PDF</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="row">
                    <input type="file" id="pdfFile" name="file" accept=".pdf" required>
                    <button type="submit" class="btn">Upload & Extract</button>
                </div>
            </form>
            <div id="uploadStatus" class="muted" style="margin-top: 10px;"></div>
        </div>

        <div id="fileInfo" class="muted"></div>

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

        <div id="resultPanel" class="panel hidden">
            <h3>Extracted Content</h3>
            <div id="extractedContent" class="result"></div>
        </div>

        <div id="searchPanel" class="panel hidden">
            <h3>Search in PDF</h3>
            <div class="row">
                <input type="text" id="searchQuery" placeholder="Enter text to search...">
                <button class="btn" onclick="searchInPDF()">Search</button>
            </div>
            <div id="searchResults" class="result"></div>
        </div>
    </div>

    <script>
        let currentFileId = null;

        document.getElementById('uploadForm').addEventListener('submit', async (event) => {
            event.preventDefault();

            const fileInput = document.getElementById('pdfFile');
            const file = fileInput.files[0];

            if (!file) {
                alert('Please select a PDF file');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('uploadStatus').textContent = 'Processing...';

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Upload failed');
                }

                currentFileId = data.file_id;
                document.getElementById('uploadStatus').textContent = 'Upload successful';
                document.getElementById('fileInfo').textContent = `File: ${data.filename} (${data.page_count} pages)`;
                document.getElementById('languageContent').textContent = JSON.stringify(data.language_profile, null, 2);
                document.getElementById('languagePanel').classList.remove('hidden');
                document.getElementById('summaryActions').classList.remove('hidden');
                document.getElementById('summaryStatus').textContent = 'Click "Summarise Whole PDF" to generate a summary from the full extracted text.';
                document.getElementById('summaryContent').textContent = '';
                document.getElementById('extractedContent').textContent = data.content;
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

                const data = await response.json();

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

                const data = await response.json();

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


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        file_id = uuid.uuid4().hex
        saved_path = upload_path(file_id, file.filename)
        file.save(saved_path)

        extracted = extract_pdf_data(saved_path)
        record = {
            "filename": file.filename,
            "filepath": str(saved_path),
            "content": extracted["content"],
            "cleaned_content": extracted["cleaned_content"],
            "pages": extracted["content_parts"],
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
                "filename": file.filename,
                "page_count": extracted["page_count"],
                "content": extracted["cleaned_content"],
                "language_profile": extracted["language_profile"],
                "structured_document": extracted["structured_document"],
                "tables": extracted["tables"],
                "metadata": extracted["metadata"],
            }
        )
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

    return jsonify(pdf_data)


@app.route("/api/structured/<file_id>", methods=["GET"])
def get_structured_content(file_id):
    pdf_data = load_record(file_id)
    if pdf_data is None:
        return jsonify({"error": "File not found"}), 404

    return jsonify(pdf_data.get("structured_document", {}))
