"""
System-wide constants. Tweak here instead of scattering magic numbers.
"""

# ── PDF Classification ─────────────────────────────────────────────────────
PDF_TYPE_DIGITAL = "digital"
PDF_TYPE_SCANNED = "scanned"
PDF_TYPE_MIXED = "mixed"

# ── Processing States ──────────────────────────────────────────────────────
STATE_PENDING = "pending"
STATE_PROCESSING = "processing"
STATE_DONE = "done"
STATE_FAILED = "failed"

# ── Layout / Sorting ───────────────────────────────────────────────────────
LINE_Y_TOLERANCE = 5  # points (digital)


def line_y_tolerance_ocr(dpi: int = 150) -> int:
    """Scale OCR line-merge tolerance proportionally to render DPI."""
    return max(2, round(8 * dpi / 300))


LINE_Y_TOLERANCE_OCR = line_y_tolerance_ocr()

DEFAULT_OCR_RENDER_DPI = 150

MIN_BLOCK_CHAR_COUNT = 3

# ── OCR Preprocessing ─────────────────────────────────────────────────────
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2
DESKEW_MAX_ANGLE = 45
MORPH_KERNEL_SIZE = (2, 2)

# ── Table Detection ───────────────────────────────────────────────────────
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2
TABLE_SCORE_THRESHOLD = 0.7

# ── Noise Removal ─────────────────────────────────────────────────────────
# FIX: was r"^[^a-zA-Z0-9\s]{3,}$" — that pattern strips entire lines of
# Gujarati/Devanagari/other Indic script because those Unicode characters are
# not in [a-zA-Z0-9]. Replaced with r"^[^\w\s]{3,}$" which uses \w (Unicode-
# aware in Python 3), so any alphanumeric Unicode letter (including gu/hi/ta)
# is excluded from the "pure-symbols" check.
NOISE_PATTERNS = [
    r"^[^\w\s]{3,}$",  # pure symbol lines (safe for all Unicode scripts)
    r"(.)\1{4,}",  # repeated same char 5+ times
    r"^\s*[-_=*#]{5,}\s*$",  # separator lines
    r"^Page\s+\d+\s+of\s+\d+$",  # page markers
]

MIN_PARA_WORD_COUNT = 2

# ── Confidence ────────────────────────────────────────────────────────────
HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.65
LOW_CONFIDENCE = 0.40

# ── Supported MIME Types ──────────────────────────────────────────────────
ALLOWED_MIME_TYPES = {"application/pdf", "application/x-pdf"}

# ── Temp File TTL ─────────────────────────────────────────────────────────
TEMP_FILE_TTL_SECONDS = 3600
