"""
Noise Removal Engine
Cleans extracted text by removing:
  - Stamps / watermarks
  - Random symbols and artifacts
  - Repeated characters
  - Near-duplicate lines
  - Header/footer boilerplate
Uses regex patterns and fuzzy deduplication.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Set

from app.config.constants import MIN_PARA_WORD_COUNT, NOISE_PATTERNS
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Pre-compile noise patterns for performance
_COMPILED_NOISE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

# Characters that appear in watermarks/stamps but not normal text
_WATERMARK_CHARS = re.compile(r"[©®™¶§†‡°•·]")

# Excessive whitespace normalizer
_MULTI_SPACE = re.compile(r" {2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _is_noise_line(line: str) -> bool:
    """Return True if the line matches any known noise pattern."""
    stripped = line.strip()
    if not stripped:
        return True  # Empty lines are not noise per se, but skip them here

    for pattern in _COMPILED_NOISE:
        if pattern.match(stripped):
            return True

    # Lines consisting only of punctuation/symbols (no alphanumeric)
    if not any(c.isalnum() for c in stripped):
        return True

    return False


def _remove_watermark_chars(text: str) -> str:
    """Remove copyright/watermark unicode symbols."""
    return _WATERMARK_CHARS.sub("", text)


def _normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters.
    NFKC: compatibility decomposition → composition.
    Converts ligatures (ﬁ→fi), curly quotes, etc.
    """
    return unicodedata.normalize("NFKC", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines."""
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def clean_line(line: str) -> str:
    """Apply all character-level cleaners to a single line."""
    line = _normalize_unicode(line)
    line = _remove_watermark_chars(line)
    line = line.strip()
    return line


def remove_noise_lines(lines: List[str]) -> List[str]:
    """
    Filter out noise lines from a list of text lines.
    Keeps lines that have sufficient alphanumeric content.
    """
    cleaned = []
    for line in lines:
        cl = clean_line(line)
        if not cl:
            continue
        if _is_noise_line(cl):
            logger.debug(f"Noise removed: '{cl[:60]}'")
            continue
        # Check minimum word count
        words = cl.split()
        if len(words) < MIN_PARA_WORD_COUNT:
            logger.debug(f"Too short, removed: '{cl}'")
            continue
        cleaned.append(cl)
    return cleaned


def _similarity(a: str, b: str) -> float:
    """Return similarity ratio between two strings (0..1)."""
    return SequenceMatcher(None, a, b).ratio()


def remove_duplicate_lines(
    lines: List[str],
    similarity_threshold: float = 0.92,
) -> List[str]:
    """
    Remove near-duplicate lines using sequence similarity.
    Maintains original order (first occurrence wins).

    similarity_threshold: 0.92 means lines must be 92%+ similar to be considered duplicate.
    """
    seen: List[str] = []
    for line in lines:
        is_dup = any(
            _similarity(line.lower(), s.lower()) >= similarity_threshold for s in seen
        )
        if not is_dup:
            seen.append(line)
        else:
            logger.debug(f"Duplicate removed: '{line[:60]}'")
    return seen


def clean_text_block(text: str) -> str:
    """
    Full cleaning pipeline for an extracted text block.

    1. Normalize unicode
    2. Split into lines
    3. Remove noise lines
    4. Remove duplicates
    5. Re-join and normalize whitespace
    """
    if not text:
        return ""

    text = _normalize_unicode(text)
    text = _remove_watermark_chars(text)

    lines = text.split("\n")
    lines = remove_noise_lines(lines)
    lines = remove_duplicate_lines(lines)

    result = "\n".join(lines)
    result = _normalize_whitespace(result)
    return result


def clean_pages(page_texts: List[str]) -> List[str]:
    """
    Apply full cleaning to each page's text.
    Also removes headers/footers that appear on every page
    (detected as lines that repeat across 60%+ of pages).
    """
    if not page_texts:
        return []

    # Find repeated lines (likely headers/footers)
    from collections import Counter

    line_counts: Counter = Counter()
    for pt in page_texts:
        for line in set(pt.split("\n")):
            if line.strip():
                line_counts[line.strip()] += 1

    total_pages = len(page_texts)
    repeated: Set[str] = {
        line
        for line, cnt in line_counts.items()
        if cnt / total_pages >= 0.6 and len(line.split()) < 12
    }

    if repeated:
        logger.info(
            f"Identified {len(repeated)} header/footer lines to remove: "
            + str([r[:40] for r in list(repeated)[:5]])
        )

    cleaned = []
    for text in page_texts:
        lines = text.split("\n")
        lines = [ln for ln in lines if ln.strip() not in repeated]
        cleaned.append(clean_text_block("\n".join(lines)))

    return cleaned
