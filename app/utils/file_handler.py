"""
File Handler Utility
Manages streaming uploads, validation, temp file lifecycle, and cleanup.
"""

import hashlib
import mimetypes
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple

import aiofiles
from fastapi import HTTPException, UploadFile

from app.config.constants import ALLOWED_MIME_TYPES, TEMP_FILE_TTL_SECONDS
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MB streaming chunk


def hash_file_sha256(path: Path) -> str:
    """Compute a SHA-256 hash for a file on disk."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_cache_dir() -> Path:
    """Directory used for cached extraction payloads keyed by file hash."""
    cache_dir = settings.OUTPUT_DIR / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(file_hash: str) -> Path:
    return get_cache_dir() / f"{file_hash}.json"


def load_cached_result(file_hash: str) -> Optional[dict]:
    """Load a cached extraction result if present and not expired."""
    cache_path = get_cache_path(file_hash)
    if not cache_path.exists():
        return None

    try:
        import json

        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    expires_at = data.get("expires_at")
    if expires_at:
        try:
            from datetime import datetime, timezone

            expiry_dt = datetime.fromisoformat(expires_at)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
            if expiry_dt <= datetime.now(timezone.utc):
                cache_path.unlink(missing_ok=True)
                return None
        except Exception:
            return None

    return data


def store_cached_result(file_hash: str, payload: dict) -> None:
    """Persist a cached extraction result keyed by file hash."""
    import json

    cache_path = get_cache_path(file_hash)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, cache_path)


async def save_upload_streaming(
    file: UploadFile,
    job_id: Optional[str] = None,
) -> Tuple[str, Path, int]:
    """
    Stream-save an uploaded file to disk.
    Returns (job_id, saved_path, file_size_bytes).

    Uses chunked I/O to handle large files (50-60 MB) without OOM.
    Raises HTTPException on validation failures.
    """
    job_id = job_id or str(uuid.uuid4())
    dest_path = settings.UPLOAD_DIR / f"{job_id}.pdf"

    # ── MIME / extension check ─────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        # browsers sometimes send "application/octet-stream" — allow and recheck below
        if content_type != "application/octet-stream":
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type: {content_type}. Only PDF allowed.",
            )

    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    # ── Streaming write ────────────────────────────────────────────────────
    total_bytes = 0
    sha256 = hashlib.sha256()

    try:
        async with aiofiles.open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                sha256.update(chunk)

                if total_bytes > settings.max_file_bytes:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds maximum allowed size of "
                            f"{settings.MAX_FILE_SIZE_MB} MB."
                        ),
                    )
                await out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        logger.error(f"Upload write failed for job {job_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # ── Magic bytes check (PDF starts with %PDF) ──────────────────────────
    with open(dest_path, "rb") as f:
        magic = f.read(4)
    if magic != b"%PDF":
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid PDF (magic bytes mismatch).",
        )

    file_hash = sha256.hexdigest()
    logger.info(
        f"Upload saved | job={job_id} | bytes={total_bytes} | sha256={file_hash}"
    )
    return job_id, dest_path, total_bytes


def get_upload_path(job_id: str) -> Path:
    """Return the path of the saved upload for a given job_id."""
    return settings.UPLOAD_DIR / f"{job_id}.pdf"


def get_output_path(job_id: str) -> Path:
    """Return the output JSON path for a given job_id."""
    return settings.OUTPUT_DIR / f"{job_id}.json"


def cleanup_job_files(job_id: str) -> None:
    """Delete temp upload and output files for a job."""
    for p in [get_upload_path(job_id), get_output_path(job_id)]:
        p.unlink(missing_ok=True)
        logger.debug(f"Cleaned up: {p}")

    hash_path = settings.UPLOAD_DIR / f"{job_id}.sha256"
    hash_path.unlink(missing_ok=True)
    logger.debug(f"Cleaned up: {hash_path}")


def purge_stale_uploads() -> int:
    """
    Remove uploads older than TEMP_FILE_TTL_SECONDS.
    Call periodically (e.g. cron or startup hook).
    Returns count of files deleted.
    """
    import time

    now = time.time()
    deleted = 0
    for f in settings.UPLOAD_DIR.glob("*.pdf"):
        age = now - f.stat().st_mtime
        if age > TEMP_FILE_TTL_SECONDS:
            f.unlink(missing_ok=True)
            deleted += 1
    logger.info(f"Purged {deleted} stale upload files.")
    return deleted
