"""
Flask API for PDF Content Extraction and Search
"""
import os
import io
from flask import Flask, request, jsonify, render_template_string
from pdf import IndianLanguagePDFExtractor
import fitz  # PyMuPDF

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store extracted data in memory (for demo purposes)
# In production, use a database
pdf_storage = {}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>PDF Content Extractor & Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .upload-section {
            border: 2px dashed #ccc;
            padding: 30px;
            text-align: center;
            margin-bottom: 20px;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        .result-section {
            margin-top: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
            display: none;
        }
        .search-section {
            margin-top: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
            display: none;
        }
        .search-box {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .search-box input {
            flex: 1;
            padding: 10px;
            font-size: 16px;
        }
        .result-content {
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
            background: white;
            padding: 15px;
            border: 1px solid #ddd;
            margin-top: 15px;
        }
        .highlight {
            background-color: yellow;
            padding: 2px;
        }
        .file-info {
            color: #666;
            margin-bottom: 10px;
        }
        .tabs {
            display: flex;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
        }
        .tab.active {
            border-bottom: 2px solid #4CAF50;
            color: #4CAF50;
            font-weight: bold;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 PDF Content Extractor & Search</h1>
        
        <div class="upload-section">
            <h3>Upload PDF File</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="pdfFile" name="file" accept=".pdf" required>
                <br><br>
                <button type="submit" class="btn">Upload & Extract</button>
            </form>
            <div id="uploadStatus"></div>
        </div>

        <div class="file-info" id="fileInfo"></div>

        <div class="tabs" id="tabs" style="display: none;">
            <button class="tab active" onclick="showTab('extract')">Extracted Content</button>
            <button class="tab" onclick="showTab('search')">Search in PDF</button>
        </div>

        <div id="extractTab" class="tab-content">
            <div class="result-section" id="resultSection">
                <h3>Extracted Content:</h3>
                <div class="result-content" id="extractedContent"></div>
            </div>
        </div>

        <div id="searchTab" class="tab-content">
            <div class="search-section" id="searchSection">
                <h3>Search in PDF</h3>
                <div class="search-box">
                    <input type="text" id="searchQuery" placeholder="Enter text to search...">
                    <button class="btn" onclick="searchInPDF()">Search</button>
                </div>
                <div class="result-content" id="searchResults"></div>
            </div>
        </div>
    </div>

    <script>
        let currentFileId = null;

        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            if (tabName === 'extract') {
                document.querySelector('.tab:nth-child(1)').classList.add('active');
                document.getElementById('extractTab').classList.add('active');
            } else {
                document.querySelector('.tab:nth-child(2)').classList.add('active');
                document.getElementById('searchTab').classList.add('active');
            }
        }

        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('pdfFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a PDF file');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('uploadStatus').innerHTML = 'Processing...';

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    currentFileId = data.file_id;
                    document.getElementById('uploadStatus').innerHTML = '✓ Upload successful!';
                    document.getElementById('fileInfo').innerHTML = `File: ${data.filename} (${data.page_count} pages)`;
                    document.getElementById('extractedContent').textContent = data.content;
                    document.getElementById('resultSection').style.display = 'block';
                    document.getElementById('searchSection').style.display = 'block';
                    document.getElementById('tabs').style.display = 'flex';
                    showTab('extract');
                } else {
                    document.getElementById('uploadStatus').innerHTML = '✗ Error: ' + data.error;
                }
            } catch (error) {
                document.getElementById('uploadStatus').innerHTML = '✗ Error: ' + error.message;
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
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        file_id: currentFileId,
                        query: query
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    if (data.results.length > 0) {
                        let resultText = `Found ${data.results.length} result(s):\\n\\n`;
                        data.results.forEach((result, index) => {
                            resultText += `--- Result ${index + 1} ---\\n`;
                            resultText += `Page: ${result.page}\\n`;
                            resultText += `Context: ${result.context}\\n\\n`;
                        });
                        document.getElementById('searchResults').textContent = resultText;
                    } else {
                        document.getElementById('searchResults').textContent = 'No matches found.';
                    }
                } else {
                    document.getElementById('searchResults').textContent = 'Error: ' + data.error;
                }
            } catch (error) {
                document.getElementById('searchResults').textContent = 'Error: ' + error.message;
            }
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Upload and extract content from PDF"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save the uploaded file
        file_id = str(len(pdf_storage) + 1)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_{file.filename}')
        file.save(filepath)
        
        # Extract text from PDF using PyMuPDF
        doc = fitz.open(filepath)
        content = []
        page_count = len(doc)
        
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()
            content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        doc.close()
        
        full_content = '\n\n'.join(content)
        
        # Store in memory
        pdf_storage[file_id] = {
            'filename': file.filename,
            'filepath': filepath,
            'content': full_content,
            'pages': content,
            'page_count': page_count
        }
        
        return jsonify({
            'file_id': file_id,
            'filename': file.filename,
            'page_count': page_count,
            'content': full_content
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search_in_pdf():
    """Search for text in uploaded PDF"""
    try:
        data = request.get_json()
        
        if not data or 'file_id' not in data or 'query' not in data:
            return jsonify({'error': 'Missing file_id or query'}), 400
        
        file_id = data['file_id']
        query = data['query'].strip()
        
        if file_id not in pdf_storage:
            return jsonify({'error': 'File not found. Please upload again.'}), 404
        
        pdf_data = pdf_storage[file_id]
        results = []
        
        # Search in each page
        for page_num, page_content in enumerate(pdf_data['pages'], 1):
            # Find all occurrences
            lower_content = page_content.lower()
            lower_query = query.lower()
            
            start = 0
            while True:
                pos = lower_content.find(lower_query, start)
                if pos == -1:
                    break
                
                # Get context around the match
                context_start = max(0, pos - 50)
                context_end = min(len(page_content), pos + len(query) + 50)
                context = page_content[context_start:context_end]
                
                results.append({
                    'page': page_num,
                    'position': pos,
                    'context': f"...{context}..."
                })
                
                start = pos + 1
        
        return jsonify({
            'query': query,
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    files = []
    for file_id, data in pdf_storage.items():
        files.append({
            'file_id': file_id,
            'filename': data['filename'],
            'page_count': data['page_count']
        })
    return jsonify({'files': files})


@app.route('/api/content/<file_id>', methods=['GET'])
def get_content(file_id):
    """Get extracted content for a specific file"""
    if file_id not in pdf_storage:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify(pdf_storage[file_id])


if __name__ == '__main__':
    app.run(debug=True, port=5000)
