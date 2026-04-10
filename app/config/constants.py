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
# Y-axis tolerance: two bounding boxes within this many pts are on the same line
LINE_Y_TOLERANCE = 5  # points  (digital)
LINE_Y_TOLERANCE_OCR = 8  # pixels  (ocr — after 300 DPI conversion)

# Minimum character count for a text block to be kept (filters noise)
MIN_BLOCK_CHAR_COUNT = 3

# ── OCR Preprocessing ─────────────────────────────────────────────────────
ADAPTIVE_BLOCK_SIZE = 11  # OpenCV adaptiveThreshold block size (odd number)
ADAPTIVE_C = 2  # Constant subtracted from mean
DESKEW_MAX_ANGLE = 45  # Degrees — beyond this we assume it's intentional rotation
MORPH_KERNEL_SIZE = (1, 1)  # Morphological operations kernel

# ── Table Detection ───────────────────────────────────────────────────────
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2
TABLE_SCORE_THRESHOLD = 0.7  # Digital table quality filter

# ── Noise Removal ─────────────────────────────────────────────────────────
# Regex patterns that indicate noise/stamps/artifacts
NOISE_PATTERNS = [
    r"^[^a-zA-Z0-9\s]{3,}$",  # Lines of pure symbols
    r"(.)\1{4,}",  # Repeated same char 5+ times
    r"^\s*[-_=*#]{5,}\s*$",  # Separator lines
    r"^Page\s+\d+\s+of\s+\d+$",  # Page markers (optional — comment out if needed)
]

# Minimum word count per extracted paragraph to survive validation
MIN_PARA_WORD_COUNT = 2

# ── Chunked Processing ────────────────────────────────────────────────────
CHUNK_SIZE_PAGES = 10  # Pages per processing chunk for large files

# ── Confidence ────────────────────────────────────────────────────────────
HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.65
LOW_CONFIDENCE = 0.40

# ── Supported MIME Types ──────────────────────────────────────────────────
ALLOWED_MIME_TYPES = {"application/pdf", "application/x-pdf"}

# ── Temp File TTL ─────────────────────────────────────────────────────────
TEMP_FILE_TTL_SECONDS = 3600  # 1 hour — after this uploads are purged
