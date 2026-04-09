"""Optional PDF OCR/extraction helpers.

This module is safe to import without the full OCR stack installed. The
primary Vercel app only needs Flask and PyMuPDF, but local users can install
the optional extras listed in the docstring below if they want OCR support.

Optional extras:
    pytesseract
    pdf2image
    PyPDF2
    Pillow
    langdetect
    opencv-python
    numpy
    fpdf
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - hard dependency for this module
    raise ImportError("PyMuPDF is required for pdf.py") from exc

warnings.filterwarnings("ignore")

try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    from langdetect import detect
except ImportError:
    pytesseract = None
    Image = None
    cv2 = None
    np = None
    detect = None


@dataclass
class ExtractionWarning:
    message: str


class IndianLanguagePDFExtractor:
    """Lightweight extractor with graceful degradation when OCR extras are absent."""

    def __init__(self, tesseract_path: Optional[str] = None):
        self.ocr_enabled = all(
            module is not None for module in (pytesseract, Image, cv2, np)
        )
        if tesseract_path and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.indian_languages = {
            "hin": {"name": "Hindi", "script": "Devanagari", "tesseract_code": "hin"},
            "ben": {"name": "Bengali", "script": "Bengali", "tesseract_code": "ben"},
            "tel": {"name": "Telugu", "script": "Telugu", "tesseract_code": "tel"},
            "tam": {"name": "Tamil", "script": "Tamil", "tesseract_code": "tam"},
            "mar": {"name": "Marathi", "script": "Devanagari", "tesseract_code": "mar"},
            "guj": {"name": "Gujarati", "script": "Gujarati", "tesseract_code": "guj"},
            "kan": {"name": "Kannada", "script": "Kannada", "tesseract_code": "kan"},
            "mal": {"name": "Malayalam", "script": "Malayalam", "tesseract_code": "mal"},
            "ori": {"name": "Odia", "script": "Odia", "tesseract_code": "ori"},
            "pan": {"name": "Punjabi", "script": "Gurmukhi", "tesseract_code": "pan"},
            "urd": {"name": "Urdu", "script": "Perso-Arabic", "tesseract_code": "urd"},
            "san": {"name": "Sanskrit", "script": "Devanagari", "tesseract_code": "san"},
            "kas": {"name": "Kashmiri", "script": "Perso-Arabic", "tesseract_code": "kas"},
            "nep": {"name": "Nepali", "script": "Devanagari", "tesseract_code": "nep"},
            "bod": {"name": "Tibetan", "script": "Tibetan", "tesseract_code": "bod"},
            "mya": {"name": "Burmese", "script": "Myanmar", "tesseract_code": "mya"},
        }

    def extract_from_pdf(
        self,
        pdf_path: str,
        languages: Optional[List[str]] = None,
        preprocessing_level: str = "medium",
        extract_images: bool = True,
        save_debug_images: bool = False,
    ) -> Dict[str, Any]:
        """Extract text from a PDF.

        If OCR extras are unavailable, direct PDF text extraction still works.
        """

        result: Dict[str, Any] = {
            "filename": os.path.basename(pdf_path),
            "extraction_date": datetime.now().isoformat(),
            "text": "",
            "pages": [],
            "metadata": {},
            "statistics": {
                "total_pages": 0,
                "pages_with_text": 0,
                "images_processed": 0,
                "languages_detected": [],
            },
        }

        if languages is None:
            languages = ["hin", "eng"]

        with fitz.open(pdf_path) as doc:
            result["metadata"] = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "page_count": len(doc),
            }
            result["statistics"]["total_pages"] = len(doc)

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                page_data = {
                    "page_number": page_num + 1,
                    "direct_text": page_text,
                    "images_text": [],
                    "has_images": False,
                    "detected_languages": self._detect_languages_in_text(page_text),
                }
                if page_text.strip():
                    result["statistics"]["pages_with_text"] += 1
                result["pages"].append(page_data)

        result["text"] = self._compile_text(result)
        if not self.ocr_enabled and extract_images:
            result["warnings"] = [
                "OCR extras are not installed; image extraction was skipped."
            ]
        return result

    def _detect_languages_in_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
        detected: List[str] = []
        if detect is not None:
            try:
                lang = detect(text)
                if lang in self.indian_languages or lang == "en":
                    detected.append(lang)
            except Exception:
                pass
        return detected

    def _compile_text(self, result: Dict[str, Any]) -> str:
        compiled = [
            "=" * 80,
            "PDF TEXT EXTRACTION REPORT",
            f"File: {result['filename']}",
            f"Date: {result['extraction_date']}",
            "=" * 80,
            "",
        ]
        compiled.append("EXTRACTION STATISTICS:")
        compiled.append(f"  Total Pages: {result['statistics']['total_pages']}")
        compiled.append(f"  Pages with Text: {result['statistics']['pages_with_text']}")
        compiled.append(f"  Images Processed: {result['statistics']['images_processed']}")
        compiled.append(
            f"  Languages Detected: {', '.join(result['statistics']['languages_detected'])}"
        )
        compiled.append("")

        for page in result["pages"]:
            compiled.append("-" * 60)
            compiled.append(f"PAGE {page['page_number']}")
            compiled.append(f"Languages detected: {', '.join(page['detected_languages'])}")
            compiled.append("-" * 60)
            if page["direct_text"].strip():
                compiled.append("")
                compiled.append(page["direct_text"].strip())
            compiled.append("")

        return "\n".join(compiled)

    def save_output(
        self,
        result: Dict[str, Any],
        output_format: str = "txt",
        output_path: Optional[str] = None,
    ) -> None:
        if not output_path:
            output_path = os.path.splitext(result["filename"])[0]

        if output_format in {"txt", "both"}:
            with open(f"{output_path}_extracted.txt", "w", encoding="utf-8") as f:
                f.write(result["text"])

        if output_format in {"json", "both"}:
            with open(f"{output_path}_extracted.json", "w", encoding="utf-8") as f:
                import json

                json.dump(result, f, ensure_ascii=False, indent=2)

    def print_languages(self) -> None:
        print("SUPPORTED INDIAN LANGUAGES")
        for code, info in self.indian_languages.items():
            print(f"{code}: {info['name']} ({info['script']})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text from PDF with Indian language support"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output file path (without extension)")
    parser.add_argument("--format", "-f", choices=["txt", "json", "both"], default="txt")
    parser.add_argument("--languages", "-l", nargs="+", default=["hin", "eng"])
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--tesseract-path")
    parser.add_argument("--list-languages", action="store_true")

    args = parser.parse_args()
    extractor = IndianLanguagePDFExtractor(args.tesseract_path)

    if args.list_languages:
        extractor.print_languages()
        return

    result = extractor.extract_from_pdf(
        pdf_path=args.pdf_path,
        languages=args.languages,
        extract_images=not args.no_images,
    )
    extractor.save_output(result, args.format, args.output)


if __name__ == "__main__":
    main()
