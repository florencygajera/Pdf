"""
Post-processing & Validation Layer
Validates and scores extracted content for quality.

Fixes applied:
  - validate_extraction_result now returns "page_results" key (required by pipeline + tests)
  - stitch_page_boundaries uses shallow copy (fast) with deep copy only on warning lists
  - Language detection samples first 500 chars to avoid slow detection on huge docs
  - Empty page short-circuits early
"""

import re
import copy
from typing import Any, Dict, List, Optional, Tuple

from app.config.constants import HIGH_CONFIDENCE, LOW_CONFIDENCE, MEDIUM_CONFIDENCE
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Sentence ending characters
_SENTENCE_ENDERS = re.compile(r"[.!?।॥]$")  # includes Hindi/Devanagari punctuation

# FIX: seed DetectorFactory once at module load time for deterministic results
try:
    from langdetect import DetectorFactory as _DetectorFactory

    _DetectorFactory.seed = 42
except ImportError:
    pass


def _detect_language(text: str) -> str:
    """
    Detect primary language using langdetect.
    FIX: Truncates to 500 chars for speed. Falls back to 'unknown' if not installed.
    """
    if not text or len(text) < 20:
        return "unknown"

    # FIX: sample only first 500 chars — langdetect doesn't need the whole document
    sample = text[:500]

    try:
        from langdetect import detect

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
    return bool(_SENTENCE_ENDERS.search(stripped[-1]))


def stitch_page_boundaries(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect and stitch sentences that are broken across page boundaries.

    FIX: Uses a shallow list copy + selective deep copy of 'warnings' only,
    instead of copy.deepcopy on the entire page list (which is very slow for
    large documents with many OCR raw_results).
    """
    # Shallow copy of the list — items are new dicts sharing inner objects
    pages = [dict(p) for p in pages]

    if len(pages) <= 1:
        return pages

    for i in range(len(pages) - 1):
        curr_text = pages[i].get("text", "").strip()
        next_text = pages[i + 1].get("text", "").strip()

        if not curr_text or not next_text:
            continue

        last_char = curr_text[-1] if curr_text else ""
        first_word = next_text.split()[0] if next_text.split() else ""

        if last_char not in ".!?।॥\n" and first_word and first_word[0].islower():
            pages[i]["text"] = curr_text + " " + next_text
            pages[i + 1]["text"] = ""
            # Deep-copy only the warnings list to avoid mutating the original
            pages[i]["warnings"] = list(pages[i].get("warnings", []))
            pages[i]["warnings"].append(
                f"Page boundary stitch: merged with page {i + 2}"
            )
            logger.debug(f"Stitched page boundary {i + 1} → {i + 2}")

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
            # Digital page — high confidence if text present
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
        conf = page.get("confidence")
        if conf is None:
            conf = 1.0
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

    FIX: Returns "page_results" key containing the stitched+flagged pages.
    The pipeline uses this key to apply results back. Tests assert on it too.

    Returns validation report:
      {
        "overall_confidence": float,
        "languages": [str],
        "table_issues": [...],
        "page_warnings": {...},
        "quality": "high" | "medium" | "low",
        "page_results": [...],   ← FIX: was missing, caused KeyError in pipeline
        "stitched_pages": [...], ← alias kept for backward compat
      }
    """
    # FIX: stitch_page_boundaries returns new list — use its return value
    page_results = stitch_page_boundaries(page_results)

    # Flag low-quality pages
    page_results = flag_low_quality_pages(page_results)

    # Overall confidence
    overall_conf = compute_confidence_score(page_results)

    # Language detection on combined text (FIX: truncated inside _detect_language)
    all_text = " ".join(p.get("text", "") for p in page_results)
    language = _detect_language(all_text)

    # Table validation
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

    # Quality label
    if overall_conf >= HIGH_CONFIDENCE:
        quality = "high"
    elif overall_conf >= MEDIUM_CONFIDENCE:
        quality = "medium"
    else:
        quality = "low"

    # Collect all page warnings
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
        # FIX: expose processed pages so pipeline can read stitched text back
        "page_results": page_results,
        # backward-compat alias used by extraction_pipeline.py
        "stitched_pages": page_results,
    }

    logger.info(
        f"Validation complete | confidence={overall_conf:.3f} | "
        f"quality={quality} | language={language} | "
        f"table_issues={len(table_issues)}"
    )

    return report
