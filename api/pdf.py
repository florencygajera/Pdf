"""Optional PDF OCR/extraction helpers with graceful degradation."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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


NOISE_PATTERNS = (
    r"^[\W_]{3,}$",
    r"^(?:[0Oo]{1,2}[\s./\\-]*){4,}$",
    r"^(?:[/\\|._-]\s*){4,}$",
    r"^(?:\.\s*){4,}$",
)


@dataclass
class ExtractionWarning:
    message: str


class IndianLanguagePDFExtractor:
    """Hybrid PDF extractor with digital text and OCR fallback."""

    def __init__(self, tesseract_path: Optional[str] = None):
        self.ocr_enabled = all(
            module is not None for module in (pytesseract, Image, cv2, np)
        )
        if tesseract_path and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        if self.ocr_enabled and pytesseract is not None:
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                self.ocr_enabled = False

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
            "eng": {"name": "English", "script": "Latin", "tesseract_code": "eng"},
        }

    def extract_from_pdf(
        self,
        pdf_path: str,
        languages: Optional[List[str]] = None,
        preprocessing_level: str = "medium",
        extract_images: bool = True,
        save_debug_images: bool = False,
    ) -> Dict[str, Any]:
        """Extract text from a PDF using digital text first and OCR as fallback."""

        result: Dict[str, Any] = {
            "filename": os.path.basename(pdf_path),
            "extraction_date": datetime.now().isoformat(),
            "text": "",
            "cleaned_text": "",
            "pages": [],
            "metadata": {},
            "statistics": {
                "total_pages": 0,
                "pages_with_text": 0,
                "images_processed": 0,
                "languages_detected": [],
                "ocr_pages": 0,
                "digital_pages": 0,
                "scanned_pages_detected": 0,
                "ocr_enabled": self.ocr_enabled,
            },
            "warnings": [],
        }

        if languages is None:
            languages = ["hin", "eng"]

        try:
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
                    "is_encrypted": bool(doc.is_encrypted),
                }
                result["statistics"]["total_pages"] = len(doc)

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_data = self._extract_page(
                        page=page,
                        page_num=page_num,
                        languages=languages,
                        preprocessing_level=preprocessing_level,
                        extract_images=extract_images,
                        save_debug_images=save_debug_images,
                        pdf_path=pdf_path,
                    )
                    result["pages"].append(page_data)
                    if page_data["cleaned_text"].strip():
                        result["statistics"]["pages_with_text"] += 1
                    if page_data["extraction_method"] == "ocr":
                        result["statistics"]["ocr_pages"] += 1
                    else:
                        result["statistics"]["digital_pages"] += 1
                    if page_data["scanned_detected"]:
                        result["statistics"]["scanned_pages_detected"] += 1
                    if page_data["raw_ocr_text"].strip():
                        result["statistics"]["images_processed"] += 1
                    if page_data.get("warning"):
                        result["warnings"].append(page_data["warning"])

        except Exception as exc:
            raise ValueError(f"Unable to process PDF: {exc}") from exc

        result["text"] = self._compile_text(result, field_name="raw_text")
        result["cleaned_text"] = self._compile_text(result, field_name="cleaned_text")
        result["statistics"]["languages_detected"] = self._collect_languages(result["pages"])
        return result

    def _extract_page(
        self,
        page,
        page_num: int,
        languages: List[str],
        preprocessing_level: str,
        extract_images: bool,
        save_debug_images: bool,
        pdf_path: str,
    ) -> Dict[str, Any]:
        direct_blocks = self._extract_digital_blocks(page)
        direct_text = "\n".join(block["text"] for block in direct_blocks).strip()
        cleaned_direct_text = self.clean_text(direct_text)
        scanned_detected = self._is_scanned_page(page, cleaned_direct_text)
        has_images = bool(page.get_images(full=True))

        raw_ocr_text = ""
        final_blocks = direct_blocks
        extraction_method = "digital"
        warning = ""

        if scanned_detected and extract_images:
            if self.ocr_enabled:
                try:
                    ocr_result = self._extract_page_with_ocr(
                        page=page,
                        languages=languages,
                        preprocessing_level=preprocessing_level,
                        save_debug_images=save_debug_images,
                        debug_prefix=f"{Path(pdf_path).stem}_page_{page_num + 1}",
                    )
                    raw_ocr_text = ocr_result["text"]
                    if raw_ocr_text.strip():
                        final_blocks = ocr_result["blocks"]
                        extraction_method = "ocr"
                    else:
                        warning = f"OCR returned no text on page {page_num + 1}; using digital fallback."
                except Exception as exc:
                    warning = f"OCR failed on page {page_num + 1}: {exc}"
            else:
                warning = (
                    f"Page {page_num + 1} appears scanned, but OCR extras are unavailable; "
                    "using direct PDF text only."
                )

        raw_text = "\n".join(block["text"] for block in final_blocks).strip()
        cleaned_text = self.clean_text(raw_text)

        if not cleaned_text and cleaned_direct_text:
            cleaned_text = cleaned_direct_text
            raw_text = direct_text
            final_blocks = direct_blocks
            extraction_method = "digital"

        return {
            "page_number": page_num + 1,
            "direct_text": direct_text,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "raw_ocr_text": raw_ocr_text,
            "images_text": [raw_ocr_text] if raw_ocr_text else [],
            "has_images": has_images,
            "detected_languages": self._detect_languages_in_text(cleaned_text),
            "scanned_detected": scanned_detected,
            "extraction_method": extraction_method,
            "blocks": final_blocks,
            "tables": self._extract_tables_from_blocks(final_blocks),
            "page_heading": self._extract_page_heading(final_blocks),
            "warning": warning,
        }

    def _extract_digital_blocks(self, page) -> List[Dict[str, Any]]:
        blocks = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            cleaned = self.clean_text(text)
            if not cleaned:
                continue
            blocks.append(
                {
                    "x0": float(x0),
                    "y0": float(y0),
                    "x1": float(x1),
                    "y1": float(y1),
                    "text": cleaned,
                }
            )
        blocks.sort(key=lambda item: (round(item["y0"], 1), item["x0"]))
        return blocks

    def _split_table_cells(self, text: str) -> List[str]:
        import re

        candidates = [cell.strip() for cell in re.split(r"\s{2,}", text) if cell.strip()]
        if len(candidates) > 1:
            return candidates
        pipe_candidates = [cell.strip() for cell in text.split("|") if cell.strip()]
        if len(pipe_candidates) > 1:
            return pipe_candidates
        return []

    def _looks_like_table_row(self, text: str) -> bool:
        import re

        cells = self._split_table_cells(text)
        if len(cells) >= 3:
            return True
        tokens = text.lower().split()
        numeric_tokens = sum(bool(re.fullmatch(r"[\d,()./-]+", token)) for token in tokens)
        return numeric_tokens >= 2 and len(tokens) >= 4

    def _extract_tables_from_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        buffer: List[str] = []

        def flush_buffer() -> None:
            nonlocal buffer
            table = self._finalize_table(buffer)
            if table is not None:
                tables.append(table)
            buffer = []

        for block in blocks:
            text = block.get("text", "").strip()
            if not text:
                continue
            if self._looks_like_table_row(text):
                buffer.append(text)
            else:
                if buffer:
                    flush_buffer()

        if buffer:
            flush_buffer()

        return tables

    def _finalize_table(self, raw_rows: List[str]) -> Optional[Dict[str, Any]]:
        parsed_rows = [self._split_table_cells(row) for row in raw_rows]
        parsed_rows = [row for row in parsed_rows if len(row) >= 2]
        if len(parsed_rows) < 2:
            return None

        width = max(len(row) for row in parsed_rows)
        if width < 2:
            return None

        parsed_rows = [row + [""] * (width - len(row)) for row in parsed_rows]
        first_row = parsed_rows[0]
        header_score = sum(
            1
            for cell in first_row
            if cell.lower().strip(":") in {"amount", "balance", "bill", "date", "description", "gross", "item", "net", "qty", "quantity", "rate", "total", "unit", "value"}
            or not cell.replace(",", "").replace(".", "").isdigit()
        )

        if header_score >= max(1, width // 2):
            columns = first_row
            rows = parsed_rows[1:]
        else:
            columns = [f"column_{index + 1}" for index in range(width)]
            rows = parsed_rows

        rows = [row for row in rows if any(cell for cell in row)]
        if not rows:
            return None

        return {"columns": columns, "rows": rows}

    def _extract_page_heading(self, blocks: List[Dict[str, Any]]) -> str:
        if not blocks:
            return ""
        first_line = (blocks[0].get("text") or "").strip()
        if not first_line:
            return ""
        if len(first_line) <= 120 and (
            first_line.isupper() or first_line.endswith(":") or len(first_line.split()) <= 8
        ):
            return first_line.rstrip(":")
        return ""

    def _extract_page_with_ocr(
        self,
        page,
        languages: List[str],
        preprocessing_level: str,
        save_debug_images: bool,
        debug_prefix: str,
    ) -> Dict[str, Any]:
        if not self.ocr_enabled:
            return {"text": "", "blocks": []}

        image = self._render_page_to_image(page)
        processed = self.preprocess_image(image, preprocessing_level)

        if save_debug_images:
            debug_path = Path(f"{debug_prefix}_preprocessed.png")
            if Image is not None:
                Image.fromarray(processed).save(debug_path)

        tesseract_lang = "+".join(
            self.indian_languages.get(language, {}).get("tesseract_code", language)
            for language in languages
        )
        config = "--oem 3 --psm 6"
        data = pytesseract.image_to_data(
            processed,
            lang=tesseract_lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        lines = {}
        count = len(data.get("text", []))
        for index in range(count):
            text = self.clean_text(data["text"][index] or "")
            if not text:
                continue

            confidence = data.get("conf", ["-1"] * count)[index]
            try:
                if float(confidence) < 0:
                    continue
            except (TypeError, ValueError):
                pass

            key = (
                data.get("block_num", [0] * count)[index],
                data.get("par_num", [0] * count)[index],
                data.get("line_num", [0] * count)[index],
            )
            entry = lines.setdefault(
                key,
                {
                    "parts": [],
                    "x0": data["left"][index],
                    "y0": data["top"][index],
                    "x1": data["left"][index] + data["width"][index],
                    "y1": data["top"][index] + data["height"][index],
                },
            )
            entry["parts"].append((data["left"][index], text))
            entry["x0"] = min(entry["x0"], data["left"][index])
            entry["y0"] = min(entry["y0"], data["top"][index])
            entry["x1"] = max(entry["x1"], data["left"][index] + data["width"][index])
            entry["y1"] = max(entry["y1"], data["top"][index] + data["height"][index])

        blocks = []
        for entry in lines.values():
            ordered_parts = [part for _, part in sorted(entry["parts"], key=lambda item: item[0])]
            line_text = self.clean_text(" ".join(ordered_parts))
            if not line_text:
                continue
            blocks.append(
                {
                    "x0": float(entry["x0"]),
                    "y0": float(entry["y0"]),
                    "x1": float(entry["x1"]),
                    "y1": float(entry["y1"]),
                    "text": line_text,
                }
            )

        blocks.sort(key=lambda item: (round(item["y0"], 1), item["x0"]))
        text = "\n".join(block["text"] for block in blocks).strip()
        return {"text": text, "blocks": blocks}

    def _render_page_to_image(self, page):
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        if np is None:
            raise RuntimeError("numpy is required for OCR image conversion")
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif pix.n == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def preprocess_image(self, image, preprocessing_level: str = "medium"):
        if not self.ocr_enabled:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if preprocessing_level == "high":
            gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)

        thresholded = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        thresholded = cv2.medianBlur(thresholded, 3)
        thresholded = self._deskew_image(thresholded)
        return thresholded

    def _deskew_image(self, image):
        if not self.ocr_enabled:
            return image

        coords = np.column_stack(np.where(image < 250))
        if coords.size == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.2:
            return image

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        lines = []
        for raw_line in text.replace("\xa0", " ").splitlines():
            line = " ".join(raw_line.split())
            line = self._strip_noise_tokens(line)
            if not line:
                continue
            line = self._fix_broken_numbers(line)
            line = self._collapse_repeated_characters(line)
            line = self._normalize_spacing(line)
            if not line or self._is_noise_line(line):
                continue
            lines.append(line)

        cleaned_lines = self._merge_wrapped_lines(lines)
        return "\n".join(cleaned_lines).strip()

    def _strip_noise_tokens(self, line: str) -> str:
        line = line.strip()
        line = line.replace("|", " | ")
        line = " ".join(line.split())
        line = line.replace("•", "-")
        return line.strip()

    def _fix_broken_numbers(self, line: str) -> str:
        line = line.replace(", ", ",")
        return __import__("re").sub(r"(?<=\d)\s+(?=\d)", "", line)

    def _collapse_repeated_characters(self, line: str) -> str:
        import re

        line = re.sub(r"([/\\|._-])\1{1,}", r"\1", line)
        line = re.sub(r"([A-Za-z])\1{4,}", r"\1", line)
        return line

    def _normalize_spacing(self, line: str) -> str:
        import re

        line = re.sub(r"\s+([,.;:])", r"\1", line)
        line = re.sub(r"([(\[])\s+", r"\1", line)
        line = re.sub(r"\s+([)\]])", r"\1", line)
        return " ".join(line.split())

    def _is_noise_line(self, text: str) -> bool:
        import re

        cleaned = text.strip()
        if not cleaned:
            return True
        if len(cleaned) == 1 and not cleaned.isalnum():
            return True
        if sum(char.isalnum() for char in cleaned) <= 2 and len(cleaned) >= 3:
            return True
        if re.search(r"(stamp|signature|seal)", cleaned, flags=re.IGNORECASE) and len(cleaned) < 24:
            return True
        return any(re.match(pattern, cleaned) for pattern in NOISE_PATTERNS)

    def _merge_wrapped_lines(self, lines: List[str]) -> List[str]:
        merged: List[str] = []
        for line in lines:
            if not merged:
                merged.append(line)
                continue

            previous = merged[-1]
            if (
                previous
                and not previous.endswith((".", ":", ";", "?", "!", "|"))
                and line
                and not line[:1].isupper()
                and "|" not in previous
                and "|" not in line
            ):
                merged[-1] = f"{previous} {line}"
            else:
                merged.append(line)
        return merged

    def _is_scanned_page(self, page, cleaned_direct_text: str) -> bool:
        alnum_count = sum(char.isalnum() for char in cleaned_direct_text)
        if alnum_count >= 40:
            return False
        if page.get_images(full=True):
            return True
        return alnum_count < 10

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

    def _collect_languages(self, pages: List[Dict[str, Any]]) -> List[str]:
        languages = []
        seen = set()
        for page in pages:
            for language in page.get("detected_languages", []):
                if language not in seen:
                    seen.add(language)
                    languages.append(language)
        return languages

    def _compile_text(self, result: Dict[str, Any], field_name: str = "cleaned_text") -> str:
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
        compiled.append(f"  OCR Pages: {result['statistics']['ocr_pages']}")
        compiled.append(f"  Digital Pages: {result['statistics']['digital_pages']}")
        compiled.append(
            f"  Languages Detected: {', '.join(result['statistics']['languages_detected'])}"
        )
        compiled.append("")

        for page in result["pages"]:
            compiled.append("-" * 60)
            compiled.append(f"PAGE {page['page_number']} ({page['extraction_method']})")
            compiled.append(f"Languages detected: {', '.join(page['detected_languages'])}")
            compiled.append("-" * 60)
            page_text = page.get(field_name, "").strip()
            if page_text:
                compiled.append("")
                compiled.append(page_text)
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
            with open(f"{output_path}_extracted.txt", "w", encoding="utf-8") as file_handle:
                file_handle.write(result["cleaned_text"])

        if output_format in {"json", "both"}:
            with open(f"{output_path}_extracted.json", "w", encoding="utf-8") as file_handle:
                import json

                json.dump(result, file_handle, ensure_ascii=False, indent=2)

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
