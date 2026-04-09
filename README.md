# PDF Content Extractor

Production-ready Flask app for uploading PDFs, extracting text, and searching within the extracted content.

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
