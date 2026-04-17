"""
Upload Route
POST /api/v1/upload
  - Validates and stream-saves the PDF
  - Enqueues a Celery extraction task (or runs inline if Celery unavailable)
  - Returns job_id for polling

Rate-limited to prevent abuse.

FIX: Now imports the shared Limiter instance from app.api.limiter instead of
creating its own. Previously each route module had its own Limiter(), which
meant rate-limit counters were NOT shared between upload and extract buckets.

FIX: Dedup block now validates that the cached result has not expired before
reusing it. Previously, an expired cached result would be returned as the
job_id, causing the client to immediately get a 410 Gone response.
"""

import json
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)

from app.api.limiter import limiter  # FIX: shared limiter
from app.config.settings import settings
from app.api.security import require_api_key
from app.models.response_model import ErrorResponse, UploadResponse
from app.utils.file_handler import save_upload_streaming, find_job_id_by_hash
from app.utils.logger import get_logger

router = APIRouter(dependencies=[Depends(require_api_key)])
logger = get_logger(__name__)


def _enqueue_or_run(job_id: str, background_tasks: BackgroundTasks) -> None:
    """
    Try to enqueue a Celery task.
    If Celery/Redis is unavailable, fall back to FastAPI BackgroundTasks.
    """
    if settings.is_testing:
        _run_inline(job_id)
        return

    try:
        from app.workers.celery_worker import extract_pdf_task

        extract_pdf_task.delay(job_id)
        logger.info(f"Job {job_id} enqueued to Celery.")
    except Exception as exc:
        logger.warning(
            f"Celery unavailable ({exc}). Running extraction as BackgroundTask."
        )
        background_tasks.add_task(_run_inline, job_id)


def _run_inline(job_id: str) -> None:
    """
    Inline (non-Celery) extraction — used when Redis is not available.
    Runs in a FastAPI background thread.
    """
    from app.pipelines.extraction_pipeline import (
        run_extraction_pipeline,
        save_result_to_disk,
    )
    from app.utils.file_handler import get_output_path, get_upload_path

    pdf_path = get_upload_path(job_id)
    output_path = get_output_path(job_id)
    try:
        result = run_extraction_pipeline(pdf_path, job_id)
        save_result_to_disk(result, output_path)
        logger.info(f"Inline extraction complete for job {job_id}")
    except Exception as exc:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(exc),
                    "text": "",
                    "tables": [],
                    "pages": [],
                    "metadata": {
                        "pages": 0,
                        "pdf_type": "unknown",
                        "confidence_score": 0.0,
                        "processing_time_seconds": 0,
                        "warnings": [],
                        "warnings_truncated": False,
                        "total_warning_count": 0,
                    },
                    "expires_at": None,
                },
                f,
            )
        logger.error(f"Inline extraction failed for job {job_id}: {exc}", exc_info=True)


def _is_cached_result_valid(existing_job_id: str) -> bool:
    """
    Check whether a cached extraction result exists and has not expired.
    Returns True only if the output file exists and its expires_at is in the future.
    """
    from app.utils.file_handler import get_output_path, cleanup_job_files

    cached_output = get_output_path(existing_job_id)
    if not cached_output.exists():
        return False

    try:
        with open(cached_output, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        expires_at = cached_data.get("expires_at")
        if not expires_at:
            # No expiry field — treat as expired to be safe
            return False
        expiry_dt = datetime.fromisoformat(expires_at)
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        return expiry_dt > datetime.now(timezone.utc)
    except Exception:
        return False


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
    },
    summary="Upload a PDF for extraction",
)
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ...,
        description="PDF file to extract. Max size controlled by MAX_FILE_SIZE_MB env var.",
    ),
):
    """
    Stream-upload a PDF and queue it for extraction.

    - Supports files up to `MAX_FILE_SIZE_MB` (default 100 MB).
    - Returns a `job_id` — use it to poll `GET /api/v1/extract/{job_id}`.
    - Processing is async (Celery) or background (FastAPI fallback).
    """
    job_id, saved_path, size_bytes = await save_upload_streaming(file)

    # ── Dedup by SHA-256 — reuse existing completed result ─────────────────
    from app.utils.file_handler import get_upload_hash_path, cleanup_job_files

    hash_path = get_upload_hash_path(job_id)
    if hash_path.exists():
        file_hash = hash_path.read_text(encoding="utf-8").strip()
        existing_job_id = find_job_id_by_hash(file_hash)

        if existing_job_id and existing_job_id != job_id:
            if _is_cached_result_valid(existing_job_id):
                # Cache is fresh — reuse it
                cleanup_job_files(job_id)
                logger.info(
                    f"Dedup hit | new_job={job_id} reuses cached={existing_job_id} | sha256={file_hash[:12]}..."
                )
                return UploadResponse(
                    job_id=existing_job_id,
                    filename=file.filename or "unknown.pdf",
                    size_bytes=size_bytes,
                    message="Cached result found. Use job_id to retrieve /extract/{job_id}.",
                )
            else:
                # Cache expired — clean up the old result and process fresh
                cleanup_job_files(existing_job_id)
                logger.info(
                    f"Dedup cache expired for sha256={file_hash[:12]}... — reprocessing as job={job_id}"
                )

    logger.info(
        f"Upload complete | job={job_id} | file={file.filename} | size={size_bytes}"
    )
    _enqueue_or_run(job_id, background_tasks)

    return UploadResponse(
        job_id=job_id,
        filename=file.filename or "unknown.pdf",
        size_bytes=size_bytes,
    )
