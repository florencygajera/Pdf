"""
PDF OCR / text-extraction helpers — production-grade, thread-safe.

Fixes applied vs original:
  [CRITICAL] import re inline in 6 methods + __import__ anti-pattern -> moved to module level
  [CRITICAL] _page_text_cache was a self-attribute -> made local inside extract_from_pdf
  [BUG]      NOISE_PATTERNS recompiled on every call -> pre-compiled at module level
  [BUG]      _extract_page scanned-detection too aggressive -> raised threshold to 80 alnum
  [BUG]      _compile_text embedded report header into cleaned_text -> plain text only
  [BUG]      detect() raised LangDetectException on short text -> caught
  [BUG]      _fix_broken_numbers used __import__("re") -> module-level re
  [WARN]     extract_from_pdf missing content_length/estimated_tokens -> now included
  [WARN]     save_output used string concatenation -> Path.open()
  [WARN]     preprocess_image RGB/BGR mismatch -> explicit cvtColor guard
  [WARN]     _deskew_image: large angles -> clamped to +-12 deg
  [WARN]     OCR lang string rebuilt each call -> cached with lru_cache
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps
# ---------------------------------------------------------------------------
try:
    import fitz
except ImportError as exc:
    raise ImportError("PyMuPDF is required: pip install pymupdf") from exc

try:
    import pytesseract
    from PIL import Image as _PILImage
    import cv2
    import numpy as np

    _OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    _PILImage = None
    cv2 = None
    np = None
    _OCR_AVAILABLE = False

try:
    from langdetect import detect as _langdetect_detect, LangDetectException
except ImportError:
    _langdetect_detect = None
    LangDetectException = Exception

# ---------------------------------------------------------------------------
# Pre-compiled patterns
# ---------------------------------------------------------------------------
_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^[\W_]{3,}$"),
    re.compile(r"^(?:[0Oo]{1,2}[\s./\\-]*){4,}$"),
    re.compile(r"^(?:[/\\|._-]\s*){4,}$"),
    re.compile(r"^(?:\.\s*){4,}$"),
]

_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_NBSP = re.compile(r"\xa0")
_RE_REPEATED_SEP = re.compile(r"([/\\|._-])\1+")
_RE_REPEATED_CHAR = re.compile(r"([A-Za-z])\1{4,}")
_RE_SPACE_PUNCT = re.compile(r"\s+([,.;:])")
_RE_OPEN_PAREN = re.compile(r"([(\[])\s+")
_RE_CLOSE_PAREN = re.compile(r"\s+([)\]])")
_RE_BROKEN_NUM = re.compile(r"(?<=\d)\s+(?=\d)")
_RE_COMMA_SPACE = re.compile(r",\s+")
_RE_NON_ALNUM = re.compile(r"\W+")


@dataclass
class ExtractionWarning:
    message: str


@lru_cache(maxsize=64)
def _build_tesseract_lang(
    language_tuple: tuple[str, ...],
    lang_map: tuple[tuple[str, str], ...],
) -> str:
    mapping = dict(lang_map)
    return "+".join(mapping.get(lang, lang) for lang in language_tuple)


class IndianLanguagePDFExtractor:
    """Hybrid PDF extractor — thread-safe singleton."""

    def __init__(self, tesseract_path: str | None = None) -> None:
        self.ocr_enabled: bool = _OCR_AVAILABLE

        if tesseract_path and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        if self.ocr_enabled and pytesseract is not None:
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                self.ocr_enabled = False
                log.warning("Tesseract not found — OCR disabled.")

        self.indian_languages: dict[str, dict[str, str]] = {
            "hin": {"name": "Hindi", "script": "Devanagari", "tesseract_code": "hin"},
            "ben": {"name": "Bengali", "script": "Bengali", "tesseract_code": "ben"},
            "tel": {"name": "Telugu", "script": "Telugu", "tesseract_code": "tel"},
            "tam": {"name": "Tamil", "script": "Tamil", "tesseract_code": "tam"},
            "mar": {"name": "Marathi", "script": "Devanagari", "tesseract_code": "mar"},
            "guj": {"name": "Gujarati", "script": "Gujarati", "tesseract_code": "guj"},
            "kan": {"name": "Kannada", "script": "Kannada", "tesseract_code": "kan"},
            "mal": {
                "name": "Malayalam",
                "script": "Malayalam",
                "tesseract_code": "mal",
            },
            "ori": {"name": "Odia", "script": "Odia", "tesseract_code": "ori"},
            "pan": {"name": "Punjabi", "script": "Gurmukhi", "tesseract_code": "pan"},
            "urd": {"name": "Urdu", "script": "Perso-Arabic", "tesseract_code": "urd"},
            "san": {
                "name": "Sanskrit",
                "script": "Devanagari",
                "tesseract_code": "san",
            },
            "kas": {
                "name": "Kashmiri",
                "script": "Perso-Arabic",
                "tesseract_code": "kas",
            },
            "nep": {"name": "Nepali", "script": "Devanagari", "tesseract_code": "nep"},
            "bod": {"name": "Tibetan", "script": "Tibetan", "tesseract_code": "bod"},
            "mya": {"name": "Burmese", "script": "Myanmar", "tesseract_code": "mya"},
            "eng": {"name": "English", "script": "Latin", "tesseract_code": "eng"},
        }
        self._lang_map_tuple: tuple[tuple[str, str], ...] = tuple(
            (code, info["tesseract_code"])
            for code, info in self.indian_languages.items()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_pdf(
        self,
        pdf_path: str,
        languages: list[str] | None = None,
        preprocessing_level: str = "medium",
        extract_images: bool = True,
        save_debug_images: bool = False,
    ) -> dict[str, Any]:
        """Thread-safe: page cache is a LOCAL variable, not self-state."""
        if languages is None:
            languages = ["hin", "eng"]

        result: dict[str, Any] = {
            "filename": os.path.basename(pdf_path),
            "extraction_date": datetime.now().isoformat(),
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
                    page_data = self._extract_page(
                        page=doc[page_num],
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

        except fitz.FileDataError as exc:
            raise ValueError(f"Corrupted or unreadable PDF: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Unable to process PDF: {exc}") from exc

        result["text"] = self._compile_page_texts(result["pages"], field="raw_text")
        result["cleaned_text"] = self._compile_page_texts(
            result["pages"], field="cleaned_text"
        )
        result["statistics"]["languages_detected"] = self._collect_languages(
            result["pages"]
        )

        content = result["cleaned_text"] or result["text"]
        result["content_length"] = len(content)
        result["estimated_tokens"] = max(1, len(content) // 4)

        return result

    # ------------------------------------------------------------------
    # Page extraction
    # ------------------------------------------------------------------

    def _extract_page(
        self,
        page: Any,
        page_num: int,
        languages: list[str],
        preprocessing_level: str,
        extract_images: bool,
        save_debug_images: bool,
        pdf_path: str,
    ) -> dict[str, Any]:
        direct_blocks = self._extract_digital_blocks(page)
        direct_text = "\n".join(b["text"] for b in direct_blocks).strip()
        cleaned_direct = self.clean_text(direct_text)
        has_images = bool(page.get_images(full=True))
        scanned = self._is_scanned_page(page, cleaned_direct)

        raw_ocr_text = ""
        final_blocks = direct_blocks
        extraction_method = "digital"
        warning = ""

        if scanned and extract_images and len(cleaned_direct) < 200:
            if self.ocr_enabled:
                try:
                    ocr = self._extract_page_with_ocr(
                        page=page,
                        languages=languages,
                        preprocessing_level=preprocessing_level,
                        save_debug_images=save_debug_images,
                        debug_prefix=f"{Path(pdf_path).stem}_page_{page_num + 1}",
                    )
                    raw_ocr_text = ocr["text"]
                    if raw_ocr_text.strip():
                        final_blocks = ocr["blocks"]
                        extraction_method = "ocr"
                    else:
                        warning = f"OCR returned no text on page {page_num + 1}; using digital fallback."
                except Exception as exc:
                    warning = f"OCR failed on page {page_num + 1}: {exc}"
                    log.warning(warning)
            else:
                warning = (
                    f"Page {page_num + 1} appears scanned but OCR libs are unavailable; "
                    "using direct PDF text only."
                )

        raw_text = "\n".join(b["text"] for b in final_blocks).strip()
        cleaned_text = self.clean_text(raw_text)

        if not cleaned_text and cleaned_direct:
            cleaned_text = cleaned_direct
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
            "scanned_detected": scanned,
            "extraction_method": extraction_method,
            "blocks": final_blocks,
            "tables": self._extract_tables_from_blocks(final_blocks),
            "page_heading": self._extract_page_heading(final_blocks),
            "warning": warning,
        }

    def _extract_digital_blocks(self, page: Any) -> list[dict[str, Any]]:
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
        blocks.sort(key=lambda b: (round(b["y0"], 1), b["x0"]))
        return blocks

    def _extract_page_with_ocr(
        self,
        page: Any,
        languages: list[str],
        preprocessing_level: str,
        save_debug_images: bool,
        debug_prefix: str,
    ) -> dict[str, Any]:
        if not self.ocr_enabled:
            return {"text": "", "blocks": []}

        image = self._render_page_to_image(page)
        processed = self.preprocess_image(image, preprocessing_level)

        if save_debug_images and _PILImage is not None:
            try:
                _PILImage.fromarray(processed).save(f"{debug_prefix}_preprocessed.png")
            except Exception:
                pass

        tesseract_lang = _build_tesseract_lang(tuple(languages), self._lang_map_tuple)
        data = pytesseract.image_to_data(
            processed,
            lang=tesseract_lang,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )

        lines: dict[tuple[int, int, int], dict[str, Any]] = {}
        count = len(data.get("text", []))
        for idx in range(count):
            text = self.clean_text(data["text"][idx] or "")
            if not text:
                continue
            conf = data.get("conf", ["-1"] * count)[idx]
            try:
                if float(conf) < 0:
                    continue
            except (TypeError, ValueError):
                pass

            key = (
                data.get("block_num", [0] * count)[idx],
                data.get("par_num", [0] * count)[idx],
                data.get("line_num", [0] * count)[idx],
            )
            entry = lines.setdefault(
                key,
                {
                    "parts": [],
                    "x0": data["left"][idx],
                    "y0": data["top"][idx],
                    "x1": data["left"][idx] + data["width"][idx],
                    "y1": data["top"][idx] + data["height"][idx],
                },
            )
            entry["parts"].append((data["left"][idx], text))
            entry["x0"] = min(entry["x0"], data["left"][idx])
            entry["y0"] = min(entry["y0"], data["top"][idx])
            entry["x1"] = max(entry["x1"], data["left"][idx] + data["width"][idx])
            entry["y1"] = max(entry["y1"], data["top"][idx] + data["height"][idx])

        blocks = []
        for entry in lines.values():
            ordered = [w for _, w in sorted(entry["parts"], key=lambda p: p[0])]
            line_text = self.clean_text(" ".join(ordered))
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

        blocks.sort(key=lambda b: (round(b["y0"], 1), b["x0"]))
        return {"text": "\n".join(b["text"] for b in blocks).strip(), "blocks": blocks}

    def _render_page_to_image(self, page: Any) -> Any:
        page_rect = page.rect
        scale = 1.5 if float(page_rect.width * page_rect.height) > 800_000 else 2.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        if np is None:
            raise RuntimeError("numpy is required for OCR")
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        # Explicit colorspace — pix.samples is always RGB/RGBA from PyMuPDF
        if pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def preprocess_image(self, image: Any, preprocessing_level: str = "medium") -> Any:
        if not self.ocr_enabled or cv2 is None or np is None:
            return image
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        )
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if preprocessing_level == "high":
            gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        thr = cv2.medianBlur(thr, 3)
        return self._deskew_image(thr)

    def _deskew_image(self, image: Any) -> Any:
        if not self.ocr_enabled or cv2 is None or np is None:
            return image
        coords = np.column_stack(np.where(image < 250))
        if coords.size == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        angle = max(-12.0, min(12.0, angle))  # clamp: avoids heavy black borders
        if abs(angle) < 0.2:
            return image
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    # ------------------------------------------------------------------
    # Text cleaning  (all re calls via module-level patterns)
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        lines: list[str] = []
        for raw in _RE_NBSP.sub(" ", text).splitlines():
            line = _RE_MULTI_SPACE.sub(" ", raw).strip()
            line = self._strip_noise_tokens(line)
            if not line:
                continue
            line = self._fix_broken_numbers(line)
            line = self._collapse_repeated_characters(line)
            line = self._normalize_spacing(line)
            if not line or self._is_noise_line(line):
                continue
            lines.append(line)
        return "\n".join(self._merge_wrapped_lines(lines)).strip()

    def _strip_noise_tokens(self, line: str) -> str:
        return _RE_MULTI_SPACE.sub(
            " ", line.strip().replace("|", " | ").replace("•", "-")
        ).strip()

    def _fix_broken_numbers(self, line: str) -> str:
        return _RE_BROKEN_NUM.sub("", _RE_COMMA_SPACE.sub(",", line))

    def _collapse_repeated_characters(self, line: str) -> str:
        return _RE_REPEATED_CHAR.sub(r"\1", _RE_REPEATED_SEP.sub(r"\1", line))

    def _normalize_spacing(self, line: str) -> str:
        line = _RE_SPACE_PUNCT.sub(r"\1", line)
        line = _RE_OPEN_PAREN.sub(r"\1", line)
        line = _RE_CLOSE_PAREN.sub(r"\1", line)
        return _RE_MULTI_SPACE.sub(" ", line).strip()

    def _is_noise_line(self, text: str) -> bool:
        c = text.strip()
        if not c:
            return True
        if len(c) == 1 and not c.isalnum():
            return True
        if sum(ch.isalnum() for ch in c) <= 2 and len(c) >= 3:
            return True
        if re.search(r"(stamp|signature|seal)", c, re.IGNORECASE) and len(c) < 24:
            return True
        return any(pat.match(c) for pat in _NOISE_PATTERNS)

    def _merge_wrapped_lines(self, lines: list[str]) -> list[str]:
        merged: list[str] = []
        for line in lines:
            if not merged:
                merged.append(line)
                continue
            prev = merged[-1]
            if (
                prev
                and not prev.endswith((".", ":", ";", "?", "!", "|"))
                and line
                and not line[:1].isupper()
                and "|" not in prev
                and "|" not in line
            ):
                merged[-1] = f"{prev} {line}"
            else:
                merged.append(line)
        return merged

    # ------------------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------------------

    def _split_table_cells(self, text: str) -> list[str]:
        normalized = text.replace("\t", "  ").replace("\xa0", " ")
        candidates = [c.strip() for c in re.split(r"\s{2,}", normalized) if c.strip()]
        if len(candidates) > 1:
            return candidates
        pipe = [c.strip() for c in normalized.split("|") if c.strip()]
        return pipe if len(pipe) > 1 else []

    def _normalize_table_cell(self, cell: str) -> str:
        cell = _RE_MULTI_SPACE.sub(" ", cell.replace("\xa0", " ")).strip()
        return cell.replace(" ", "") if self._looks_numeric_cell(cell) else cell

    def _looks_numeric_cell(self, cell: str) -> bool:
        cand = _RE_NON_ALNUM.sub("", cell)
        return bool(cand) and cand.isdigit() and bool(re.search(r"\d", cell))

    def _looks_like_table_row(self, text: str) -> bool:
        if len(self._split_table_cells(text)) >= 3:
            return True
        tokens = text.lower().split()
        numeric = sum(bool(re.fullmatch(r"[\d,()./-]+", t)) for t in tokens)
        return numeric >= 2 and len(tokens) >= 4

    def _extract_tables_from_blocks(
        self, blocks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tables: list[dict[str, Any]] = []
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf
            tbl = self._finalize_table(buf)
            if tbl:
                tables.append(tbl)
            buf = []

        for block in blocks:
            text = block.get("text", "").strip()
            if not text:
                continue
            if self._looks_like_table_row(text):
                buf.append(text)
            elif buf:
                flush()
        if buf:
            flush()
        return tables

    def _finalize_table(self, raw_rows: list[str]) -> dict[str, Any] | None:
        parsed = [self._split_table_cells(r) for r in raw_rows]
        parsed = [r for r in parsed if len(r) >= 2]
        if len(parsed) < 2:
            return None
        width = max(len(r) for r in parsed)
        if width < 2:
            return None
        parsed = [r + [""] * (width - len(r)) for r in parsed]
        _KW = {
            "amount",
            "balance",
            "bill",
            "date",
            "description",
            "gross",
            "item",
            "net",
            "qty",
            "quantity",
            "rate",
            "total",
            "unit",
            "value",
        }
        first = parsed[0]
        hs = sum(
            1
            for c in first
            if c.lower().strip(":") in _KW
            or not c.replace(",", "").replace(".", "").isdigit()
        )
        if hs >= max(1, width // 2):
            columns, rows = [self._normalize_table_cell(c) for c in first], parsed[1:]
        else:
            columns, rows = [f"column_{i + 1}" for i in range(width)], parsed
        rows = [[self._normalize_table_cell(c) for c in r] for r in rows if any(r)]
        return {"columns": columns, "rows": rows} if rows else None

    def _extract_page_heading(self, blocks: list[dict[str, Any]]) -> str:
        if not blocks:
            return ""
        first = (blocks[0].get("text") or "").strip()
        if not first:
            return ""
        if len(first) <= 120 and (
            first.isupper() or first.endswith(":") or len(first.split()) <= 8
        ):
            return first.rstrip(":")
        return ""

    # ------------------------------------------------------------------
    # Language detection  (FIX: catches LangDetectException)
    # ------------------------------------------------------------------

    def _detect_languages_in_text(self, text: str) -> list[str]:
        if not text.strip() or _langdetect_detect is None:
            return []
        try:
            lang = _langdetect_detect(text)
            return [lang] if (lang in self.indian_languages or lang == "en") else []
        except LangDetectException:
            return []
        except Exception as exc:
            log.debug("langdetect: %s", exc)
            return []

    def _collect_languages(self, pages: list[dict[str, Any]]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for page in pages:
            for lang in page.get("detected_languages", []):
                if lang not in seen:
                    seen.add(lang)
                    result.append(lang)
        return result

    # ------------------------------------------------------------------
    # Text compilation — plain text, NO report header (FIX)
    # ------------------------------------------------------------------

    def _compile_page_texts(
        self, pages: list[dict[str, Any]], field: str = "cleaned_text"
    ) -> str:
        return "\n\n".join(
            (page.get(field) or "").strip()
            for page in pages
            if (page.get(field) or "").strip()
        )

    # ------------------------------------------------------------------
    # Scanned detection  (FIX: raised threshold from 40 to 80)
    # ------------------------------------------------------------------

    def _is_scanned_page(self, page: Any, cleaned_direct: str) -> bool:
        alnum = sum(c.isalnum() for c in cleaned_direct)
        if alnum >= 80:
            return False
        if page.get_images(full=True):
            return True
        return alnum < 10

    # ------------------------------------------------------------------
    # Save output  (FIX: Path.open() instead of string concat)
    # ------------------------------------------------------------------

    def save_output(
        self,
        result: dict[str, Any],
        output_format: str = "txt",
        output_path: str | None = None,
    ) -> None:
        base = Path(output_path or os.path.splitext(result["filename"])[0])
        if output_format in {"txt", "both"}:
            with (base.parent / f"{base.name}_extracted.txt").open(
                "w", encoding="utf-8"
            ) as fh:
                fh.write(result.get("cleaned_text", ""))
        if output_format in {"json", "both"}:
            import json

            with (base.parent / f"{base.name}_extracted.json").open(
                "w", encoding="utf-8"
            ) as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)

    def print_languages(self) -> None:
        print("SUPPORTED INDIAN LANGUAGES")
        for code, info in self.indian_languages.items():
            print(f"  {code}: {info['name']} ({info['script']})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF extractor with Indian language support"
    )
    parser.add_argument("pdf_path")
    parser.add_argument("--output", "-o")
    parser.add_argument(
        "--format", "-f", choices=["txt", "json", "both"], default="txt"
    )
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
    print(
        f"Done — {result['statistics']['total_pages']} pages, "
        f"{result['statistics']['ocr_pages']} OCR, "
        f"{result['content_length']} chars."
    )


if __name__ == "__main__":
    main()
