"""
Service-level noise cleaner — explicit re-export (S1 fix: no wildcard import).
"""

from app.utils.noise_cleaner import (  # noqa: F401
    clean_line,
    clean_pages,
    clean_text_block,
    remove_duplicate_lines,
    remove_noise_lines,
)
