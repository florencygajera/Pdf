"""Flask API for PDF content extraction, cleanup, and structured parsing."""

import json
import re
import tempfile
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename


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

HEADING_KEYWORDS = {
    "abstract",
    "annexure",
    "amount",
    "bill",
    "certificate",
    "deductions",
    "details",
    "engineering",
    "financial",
    "gross amount",
    "introduction",
    "invoice",
    "net payable",
    "notes",
    "particulars",
    "payment",
    "railway",
    "schedule",
    "section",
    "statement",
    "summary",
    "tax",
    "total",
}

TABLE_HEADER_HINTS = {
    "amount",
    "balance",
    "bill",
    "code",
    "date",
    "deduction",
    "description",
    "gross",
    "item",
    "net",
    "no",
    "particular",
    "payable",
    "qty",
    "quantity",
    "rate",
    "remarks",
    "serial",
    "sr",
    "total",
    "unit",
    "value",
}

NOISE_PATTERNS = [
    re.compile(r"^[\W_]{3,}$"),
    re.compile(r"^(?:[0Oo]{1,2}[\s./\\-]*){4,}$"),
    re.compile(r"^(?:[/\\|._-]\s*){4,}$"),
    re.compile(r"^(?:\.\s*){4,}$"),
]


def record_path(file_id: str) -> Path:
    return CACHE_DIR / f"{file_id}.json"


def upload_path(file_id: str, filename: str) -> Path:
    safe_name = secure_filename(filename) or "upload.pdf"
    return UPLOAD_DIR / f"{file_id}_{safe_name}"


def save_record(file_id: str, record: dict) -> None:
    pdf_storage[file_id] = record
    record_path(file_id).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


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
    return re.sub(r"[ \t]+", " ", text.replace("\xa0", " ")).strip()


def collapse_broken_words(text: str) -> str:
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)", "", text)
    return normalize_whitespace(text)


def is_noise_line(text: str) -> bool:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return True
    if len(cleaned) == 1 and not cleaned.isalnum():
        return True
    if re.search(r"(stamp|signature|seal)", cleaned, flags=re.IGNORECASE) and len(cleaned) < 24:
        return True
    if sum(char.isalnum() for char in cleaned) <= 2 and len(cleaned) >= 3:
        return True
    if len(re.findall(r"[^A-Za-z0-9\u0900-\u097F\u0A80-\u0AFF\s]", cleaned)) > max(6, len(cleaned) * 0.45):
        return True
    return any(pattern.match(cleaned) for pattern in NOISE_PATTERNS)


def clean_line(text: str) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = re.sub(r"([/\\|._-])\1{2,}", r"\1", cleaned)
    cleaned = re.sub(r"([A-Za-z])\1{4,}", r"\1", cleaned)
    cleaned = collapse_broken_words(cleaned)
    return cleaned


def is_heading(text: str) -> bool:
    cleaned = normalize_whitespace(text).rstrip(":")
    if not cleaned or len(cleaned) > 120:
        return False
    lowered = cleaned.lower()
    words = lowered.split()
    if len(words) > 14:
        return False
    if lowered in HEADING_KEYWORDS:
        return True
    if any(keyword in lowered for keyword in HEADING_KEYWORDS) and len(words) <= 8:
        return True
    if cleaned.isupper() and any(char.isalpha() for char in cleaned):
        return True
    alpha_words = [word for word in re.findall(r"[A-Za-z]+", cleaned)]
    if alpha_words and sum(word[:1].isupper() for word in alpha_words) >= max(1, len(alpha_words) - 1):
        return len(alpha_words) <= 8
    return cleaned.endswith(":")


def split_table_cells(text: str) -> list[str]:
    candidates = [normalize_whitespace(cell) for cell in re.split(r"\s{2,}", text) if normalize_whitespace(cell)]
    if len(candidates) > 1:
        return candidates
    pipe_candidates = [normalize_whitespace(cell) for cell in text.split("|") if normalize_whitespace(cell)]
    if len(pipe_candidates) > 1:
        return pipe_candidates
    return []


def looks_like_table_row(text: str) -> bool:
    cells = split_table_cells(text)
    if len(cells) >= 3:
        return True
    tokens = text.lower().split()
    numeric_tokens = sum(bool(re.fullmatch(r"[\d,()./-]+", token)) for token in tokens)
    has_header_hint = sum(token.strip(":") in TABLE_HEADER_HINTS for token in tokens)
    return numeric_tokens >= 2 and (len(tokens) >= 4 or has_header_hint >= 1)


def pad_rows(rows: list[list[str]], width: int) -> list[list[str]]:
    return [row + [""] * (width - len(row)) for row in rows]


def finalize_table(raw_rows: list[str]) -> dict | None:
    parsed_rows = [split_table_cells(row) for row in raw_rows]
    parsed_rows = [row for row in parsed_rows if len(row) >= 2]
    if len(parsed_rows) < 2:
        return None

    width = max(len(row) for row in parsed_rows)
    if width < 2:
        return None

    parsed_rows = pad_rows(parsed_rows, width)
    first_row = parsed_rows[0]
    header_score = sum(
        cell.lower().strip(":") in TABLE_HEADER_HINTS or not re.fullmatch(r"[\d,()./-]+", cell or "")
        for cell in first_row
        if cell
    )

    if header_score >= max(1, width // 2):
        columns = first_row
        data_rows = parsed_rows[1:]
    else:
        columns = [f"column_{index + 1}" for index in range(width)]
        data_rows = parsed_rows

    data_rows = [row for row in data_rows if any(cell for cell in row)]
    if not data_rows:
        return None

    return {"columns": columns, "rows": data_rows}


def extract_page_lines(page) -> list[dict]:
    blocks = page.get_text("blocks")
    lines = []
    if blocks:
        for block in blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            cleaned = clean_line(text)
            if is_noise_line(cleaned):
                continue
            lines.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": cleaned})
    else:
        for index, raw_line in enumerate(page.get_text("text").splitlines()):
            cleaned = clean_line(raw_line)
            if is_noise_line(cleaned):
                continue
            lines.append({"x0": 0, "y0": float(index), "x1": 0, "y1": float(index), "text": cleaned})

    lines.sort(key=lambda item: (round(item["y0"], 1), item["x0"]))
    return lines


def lines_to_sections(page_lines: list[dict]) -> list[dict]:
    sections = []
    current_section = {"heading": "General", "content_lines": [], "tables": []}
    table_buffer: list[str] = []

    def flush_table() -> None:
        nonlocal table_buffer, current_section
        table = finalize_table(table_buffer)
        if table is not None:
            current_section["tables"].append(table)
        else:
            current_section["content_lines"].extend(table_buffer)
        table_buffer = []

    def flush_section() -> None:
        nonlocal current_section
        flush_table()
        content = "\n".join(current_section["content_lines"]).strip()
        if content or current_section["tables"] or current_section["heading"] != "General":
            sections.append(
                {
                    "heading": current_section["heading"],
                    "content": content,
                    "tables": current_section["tables"],
                }
            )
        current_section = {"heading": "General", "content_lines": [], "tables": []}

    for line in page_lines:
        text = line["text"]
        if is_heading(text):
            flush_section()
            current_section["heading"] = text.rstrip(":")
            continue

        if looks_like_table_row(text):
            table_buffer.append(text)
            continue

        if table_buffer:
            flush_table()

        current_section["content_lines"].append(text)

    flush_section()
    return sections


def merge_duplicate_sections(sections: list[dict]) -> list[dict]:
    merged = []
    index_by_heading = {}
    for section in sections:
        heading = section["heading"].strip() or "General"
        if heading not in index_by_heading:
            index_by_heading[heading] = len(merged)
            merged.append(
                {
                    "heading": heading,
                    "content": section["content"].strip(),
                    "tables": list(section["tables"]),
                }
            )
            continue

        target = merged[index_by_heading[heading]]
        if section["content"]:
            if target["content"]:
                target["content"] = f"{target['content']}\n{section['content']}".strip()
            else:
                target["content"] = section["content"].strip()
        target["tables"].extend(section["tables"])
    return merged


def extract_document_title(doc, page_lines: list[dict]) -> str:
    metadata_title = normalize_whitespace(doc.metadata.get("title", ""))
    if metadata_title:
        return metadata_title
    top_lines = [line["text"] for line in page_lines[:5] if len(line["text"]) > 3]
    return top_lines[0] if top_lines else "Untitled Document"


def structured_sections_to_text(sections: list[dict]) -> str:
    parts = []
    for section in sections:
        parts.append(section["heading"])
        if section["content"]:
            parts.append(section["content"])
        for table in section["tables"]:
            header = " | ".join(table["columns"])
            parts.append(header)
            for row in table["rows"]:
                parts.append(" | ".join(row))
    return "\n\n".join(part for part in parts if part).strip()


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
    content_parts = []
    page_texts = []
    structured_sections = []
    document_title = ""
    with fitz.open(str(saved_path)) as doc:
        page_count = len(doc)
        for page_num in range(page_count):
            page = doc[page_num]
            page_lines = extract_page_lines(page)
            if page_num == 0:
                document_title = extract_document_title(doc, page_lines)
            page_sections = lines_to_sections(page_lines)
            structured_sections.extend(page_sections)
            page_text = structured_sections_to_text(page_sections)
            page_texts.append(page_text)
            content_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

    structured_sections = merge_duplicate_sections(structured_sections)
    full_content = "\n\n".join(content_parts).strip()
    language_profile = detect_languages(full_content)
    structured_document = {
        "document_title": document_title,
        "sections": structured_sections,
    }

    return {
        "page_count": page_count,
        "full_content": full_content,
        "page_texts": page_texts,
        "language_profile": language_profile,
        "content_parts": content_parts,
        "structured_document": structured_document,
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
            "content": extracted["full_content"],
            "pages": extracted["content_parts"],
            "page_texts": extracted["page_texts"],
            "language_profile": extracted["language_profile"],
            "page_count": extracted["page_count"],
            "structured_document": extracted["structured_document"],
        }
        save_record(file_id, record)

        return jsonify(
            {
                "file_id": file_id,
                "filename": file.filename,
                "page_count": extracted["page_count"],
                "content": extracted["full_content"],
                "language_profile": extracted["language_profile"],
                "structured_document": extracted["structured_document"],
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

        results = []
        for page_num, page_content in enumerate(pdf_data["pages"], start=1):
            lower_content = page_content.lower()
            lower_query = query.lower()
            start = 0

            while True:
                pos = lower_content.find(lower_query, start)
                if pos == -1:
                    break

                context_start = max(0, pos - 50)
                context_end = min(len(page_content), pos + len(query) + 50)
                context = page_content[context_start:context_end]
                results.append(
                    {
                        "page": page_num,
                        "position": pos,
                        "context": f"...{context}...",
                    }
                )
                start = pos + 1

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
            pdf_data["content"],
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

