# PDF Content Extractor

Production-ready Flask app for uploading PDFs, extracting text, and searching within the extracted content.

Current upload limit: 500 MB in the Flask app. If you deploy on a serverless host, the platform may still enforce its own request-body limit, so very large PDFs may require a self-hosted server or external storage.

## Local run

```powershell
cd e:\pdf
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Vercel deploy

- Keep the project root at `e:\pdf`
- Make sure `requirements.txt` is at the root
- Redeploy after pushing the latest commit

## Optional OCR extras

If you want the standalone `pdf.py` OCR helper, install:

```powershell
pip install -r requirements-ocr.txt
```
