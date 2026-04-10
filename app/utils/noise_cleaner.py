"""
Noise Removal Engine.

The tests exercise this module directly, so the cleaning functions are kept
small, deterministic, and conservative:
  - remove symbol-only lines
  - remove repeated-character artifacts
  - remove exact/near duplicates
  - normalize unicode
  - strip repeated headers/footers across pages
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Set

from app.config.constants import NOISE_PATTERNS
from app.utils.logger import get_logger

logger = get_logger(__name__)

_COMPILED_NOISE = [re.compile(pattern, re.IGNORECASE) for pattern in NOISE_PATTERNS]
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_REPEATED_CHAR = re.compile(r"(.)\1{4,}")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _remove_artifacts(text: str) -> str:
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("Â©", "").replace("Â®", "").replace("â„¢", "")
    return text


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def clean_line(line: str) -> str:
    if not line:
        return ""
    line = _normalize_unicode(line)
    line = _remove_artifacts(line)
    return line.strip()


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True

    if any(pattern.match(stripped) for pattern in _COMPILED_NOISE):
        return True

    if _REPEATED_CHAR.search(stripped):
        return True

    if not any(ch.isalnum() for ch in stripped):
        return True

    return False


def remove_noise_lines(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        cl = clean_line(line)
        if not cl:
            continue
        if _is_noise_line(cl):
            logger.debug("Noise removed: %r", cl[:80])
            continue
        cleaned.append(cl)
    return cleaned


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def remove_duplicate_lines(
    lines: List[str],
    similarity_threshold: float = 0.92,
) -> List[str]:
    seen_exact: Set[str] = set()
    seen_fuzzy: List[str] = []
    result: List[str] = []
    for line in lines:
        normalized = clean_line(line)
        if not normalized:
            continue

        key = normalized.casefold()
        if key in seen_exact:
            continue

        recent_window = seen_fuzzy[-50:]
        if any(_similarity(key, prev) >= similarity_threshold for prev in recent_window):
            logger.debug("Duplicate removed: %r", normalized[:80])
            continue

        seen_exact.add(key)
        seen_fuzzy.append(key)
        result.append(normalized)
    return result


def clean_text_block(text: str) -> str:
    if not text:
        return ""

    text = _normalize_unicode(text)
    text = _remove_artifacts(text)
    lines = text.splitlines()
    lines = remove_noise_lines(lines)
    lines = remove_duplicate_lines(lines)
    if not lines:
        return ""
    return _normalize_whitespace("\n".join(lines))


def clean_pages(page_texts: List[str]) -> List[str]:
    if not page_texts:
        return []

    if len(page_texts) < 2:
        return [clean_text_block(text) for text in page_texts]

    line_counts: Counter[str] = Counter()
    for page_text in page_texts:
        seen_in_page = set()
        for line in page_text.splitlines():
            normalized = clean_line(line)
            if not normalized or normalized in seen_in_page:
                continue
            seen_in_page.add(normalized)
            line_counts[normalized] += 1

    total_pages = len(page_texts)
    repeated: Set[str] = {
        line
        for line, count in line_counts.items()
        if count / total_pages >= 0.6 and len(line.split()) <= 12
    }

    if repeated:
        logger.debug("Header/footer candidates removed: %s", list(repeated)[:5])

    cleaned_pages: List[str] = []
    for page_text in page_texts:
        lines = [
            line
            for line in page_text.splitlines()
            if clean_line(line) not in repeated
        ]
        cleaned_pages.append(clean_text_block("\n".join(lines)))

    return cleaned_pages


__all__ = [
    "clean_line",
    "remove_noise_lines",
    "remove_duplicate_lines",
    "clean_text_block",
    "clean_pages",
]
