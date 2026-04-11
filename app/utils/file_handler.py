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
    hash_path = dest_path.with_suffix(".sha256")
    hash_path.write_text(file_hash, encoding="utf-8")
    logger.info(
        f"Upload saved | job={job_id} | bytes={total_bytes} | sha256={file_hash}"
    )
    return job_id, dest_path, total_bytes


def get_upload_path(job_id: str) -> Path:
    """Return the path of the saved upload for a given job_id."""
    return settings.UPLOAD_DIR / f"{job_id}.pdf"


def get_upload_hash_path(job_id: str) -> Path:
    """Return the path of the SHA-256 file for a given upload."""
    return settings.UPLOAD_DIR / f"{job_id}.sha256"


def get_output_path(job_id: str) -> Path:
    """Return the output JSON path for a given job_id."""
    return settings.OUTPUT_DIR / f"{job_id}.json"


def find_job_id_by_hash(file_hash: str) -> Optional[str]:
    """
    Scan existing .sha256 sidecar files to find a completed job for this hash.
    Returns the job_id string if a matching completed result exists, else None.
    """
    for hash_file in settings.UPLOAD_DIR.glob("*.sha256"):
        try:
            if hash_file.read_text(encoding="utf-8").strip() == file_hash:
                job_id = hash_file.stem
                output = get_output_path(job_id)
                if output.exists():
                    return job_id
        except Exception:
            continue
    return None


def cleanup_job_files(job_id: str) -> None:
    """Delete temp upload and output files for a job."""
    for p in [
        get_upload_path(job_id),
        get_upload_hash_path(job_id),
        get_output_path(job_id),
    ]:
        p.unlink(missing_ok=True)
        logger.debug(f"Cleaned up: {p}")


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
            f.with_suffix(".sha256").unlink(missing_ok=True)
            deleted += 1
    logger.info(f"Purged {deleted} stale upload files.")
    return deleted
