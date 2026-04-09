# PDF Content Extractor

Production-ready Flask app for uploading PDFs, extracting text, and searching within the extracted content.

Upload flow:
- Paste a cloud PDF URL directly into the app, or
- Choose `Upload from Device` to send the file directly to Cloudinary from the browser, then pass the resulting `secure_url` to Flask.

The Flask app accepts only `{"file_url":"https://..."}` at `/api/upload`, streams the file temporarily for extraction, and returns only a small preview. The remote download is capped for Vercel-safe processing.

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
- Set these environment variables for device uploads:
  - `CLOUDINARY_CLOUD_NAME`
  - `CLOUDINARY_UPLOAD_PRESET`
  - Use an unsigned preset that allows `raw` uploads for PDF files
- If you split the frontend and API onto different origins later, set `CORS_ALLOWED_ORIGINS` to a comma-separated list of allowed origins.

## Upload limits

- Device uploads are limited to 50 MB on the client side before Cloudinary upload starts.
- Flask never receives the raw device file.
- URL uploads stay JSON-only and never send multipart form data to the backend.
- On Vercel, search and summary requests may land on a different serverless instance than the upload request. If that happens, re-upload the PDF before searching or summarizing.

## Optional OCR extras

If you want the standalone `pdf.py` OCR helper, install:

```powershell
pip install -r requirements-ocr.txt
```
