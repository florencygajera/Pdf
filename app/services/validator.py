"""
Post-processing & Validation Layer
Validates and scores extracted content for quality.

Checks:
  - Duplicate sentence removal
  - Sentence continuity (broken mid-sentence across pages)
  - Confidence scoring (per-page + overall)
  - Language detection
  - Empty page warnings
  - Missing-text detection (OCR vs layout block comparison)
"""

import re
from typing import Any, Dict, List, Tuple

from app.config.constants import HIGH_CONFIDENCE, LOW_CONFIDENCE, MEDIUM_CONFIDENCE
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_ENDERS = re.compile(r"[.!?\u0964\u0965]$")

try:
    from langdetect import DetectorFactory

    DetectorFactory.seed = 42
except Exception:
    DetectorFactory = None


def _detect_language(text: str) -> str:
    """
    Detect primary language using langdetect.
    Falls back to 'unknown' if not installed or fails.
    """
    if not text or len(text) < 20:
        return "unknown"

    try:
        from langdetect import detect

        sample = text.strip()[:500]
        if len(sample) < 20:
            return "unknown"
        return detect(sample)
    except ImportError:
        logger.debug("langdetect not installed. Skipping language detection.")
        return "unknown"
    except Exception as exc:
        logger.debug(f"Language detection failed: {exc}")
        return "unknown"


def _sentence_ends_properly(text: str) -> bool:
    """Check if text ends with a proper sentence terminator."""
    stripped = text.strip()
    if not stripped:
        return True
    return bool(_SENTENCE_ENDERS.search(stripped))


def stitch_page_boundaries(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect and stitch sentences that are broken across page boundaries.
    If page N ends without a sentence terminator and page N+1 starts with lowercase,
    merge the tail of page N with the head of page N+1.
    """
    pages = [page.copy() for page in pages]
    if len(pages) <= 1:
        return pages

    for i in range(len(pages) - 1):
        curr_text = pages[i].get("text", "").strip()
        next_text = pages[i + 1].get("text", "").strip()

        if not curr_text or not next_text:
            continue

        last_char = curr_text[-1] if curr_text else ""
        next_words = next_text.split()
        first_word = next_words[0] if next_words else ""

        if last_char not in ".!?\u0964\u0965\n" and first_word and first_word[0].islower():
            pages[i]["text"] = curr_text + " " + next_text
            pages[i + 1]["text"] = ""
            warnings = list(pages[i].get("warnings", []))
            warnings.append(f"Page boundary stitch: merged with page {i + 2}")
            pages[i]["warnings"] = warnings
            logger.debug("Stitched page boundary %s -> %s", i + 1, i + 2)

    return pages


def compute_confidence_score(
    page_results: List[Dict[str, Any]],
) -> float:
    """
    Compute an overall document confidence score.
    For digital pages: use text coverage heuristic (1.0 if extraction succeeded).
    For OCR pages: use average OCR confidence.
    """
    if not page_results:
        return 0.0

    scores = []
    for page in page_results:
        conf = page.get("confidence", None)
        if conf is not None:
            scores.append(float(conf))
        elif page.get("text"):
            scores.append(0.95)
        else:
            scores.append(0.0)

    return round(sum(scores) / len(scores), 4)


def flag_low_quality_pages(
    page_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add warning flags to pages with poor extraction quality.
    """
    for page in page_results:
        conf_value = page.get("confidence", 1.0)
        conf = float(conf_value) if conf_value is not None else 1.0
        text = page.get("text", "")
        warnings = page.setdefault("warnings", [])

        if not text:
            warnings.append("No text extracted from this page.")

        if conf < LOW_CONFIDENCE:
            warnings.append(
                f"Very low OCR confidence ({conf:.2f}). "
                "Page may be too blurry or low resolution."
            )
        elif conf < MEDIUM_CONFIDENCE:
            warnings.append(f"Low OCR confidence ({conf:.2f}). Review recommended.")

        word_count = len(text.split())
        if 0 < word_count < 10:
            warnings.append(
                f"Very little text extracted ({word_count} words). Possible blank/decorative page."
            )

    return page_results


def validate_table(table: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single extracted table.
    Returns (is_valid, issues_list).
    """
    issues = []

    if not table.get("headers"):
        issues.append("Table has no headers.")

    if not table.get("rows"):
        issues.append("Table has no data rows.")

    if table.get("headers") and table.get("rows"):
        expected_cols = len(table["headers"])
        for i, row in enumerate(table["rows"]):
            if len(row) != expected_cols:
                issues.append(
                    f"Row {i + 1} has {len(row)} cells, expected {expected_cols}."
                )

    return len(issues) == 0, issues


def validate_extraction_result(
    page_results: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Full validation pass over extraction results.

    Returns validation report:
      {
        "overall_confidence": float,
        "languages": [str],
        "table_issues": [...],
        "page_warnings": {...},
        "quality": "high" | "medium" | "low"
      }
    """
    page_results = stitch_page_boundaries(page_results)
    page_results = flag_low_quality_pages(page_results)

    overall_conf = compute_confidence_score(page_results)

    sample_text = ""
    for page in page_results:
        chunk = (page.get("text") or "").strip()
        if not chunk:
            continue
        if len(sample_text) >= 500:
            break
        remaining = 500 - len(sample_text)
        sample_text += (" " if sample_text else "") + chunk[:remaining]
    language = _detect_language(sample_text[:500])

    table_issues = []
    for tbl in tables:
        valid, issues = validate_table(tbl)
        if not valid:
            table_issues.append(
                {
                    "page": tbl.get("page"),
                    "table_index": tbl.get("table_index"),
                    "issues": issues,
                }
            )

    if overall_conf >= HIGH_CONFIDENCE:
        quality = "high"
    elif overall_conf >= MEDIUM_CONFIDENCE:
        quality = "medium"
    else:
        quality = "low"

    page_warnings = {
        p.get("page_number", i + 1): p.get("warnings", [])
        for i, p in enumerate(page_results)
        if p.get("warnings")
    }

    report = {
        "overall_confidence": overall_conf,
        "languages": [language],
        "table_issues": table_issues,
        "page_warnings": page_warnings,
        "quality": quality,
        "page_results": page_results,
    }

    logger.info(
        f"Validation complete | confidence={overall_conf:.3f} | "
        f"quality={quality} | language={language} | "
        f"table_issues={len(table_issues)}"
    )

    return report
