"""Flask API for PDF Content Extraction and Search."""

import json
import tempfile
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

STORAGE_DIR = Path(tempfile.gettempdir()) / "pdf_content_extractor"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CACHE_DIR = STORAGE_DIR / "cache"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for warm invocations.
pdf_storage = {}


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
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


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

        content_parts = []
        with fitz.open(str(saved_path)) as doc:
            page_count = len(doc)
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()
                content_parts.append(f"--- Page {page_num + 1} ---\n{text}")

        full_content = "\n\n".join(content_parts)
        record = {
            "filename": file.filename,
            "filepath": str(saved_path),
            "content": full_content,
            "pages": content_parts,
            "page_count": page_count,
        }
        save_record(file_id, record)

        return jsonify(
            {
                "file_id": file_id,
                "filename": file.filename,
                "page_count": page_count,
                "content": full_content,
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
