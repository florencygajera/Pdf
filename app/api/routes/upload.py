"""
Upload Route
POST /api/v1/upload
  - Validates and stream-saves the PDF
  - Enqueues a Celery extraction task (or runs inline if Celery unavailable)
  - Returns job_id for polling

Rate-limited to prevent abuse.
"""

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config.settings import settings
from app.models.response_model import ErrorResponse, UploadResponse
from app.utils.file_handler import save_upload_streaming
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)


def _enqueue_or_run(job_id: str, background_tasks: BackgroundTasks) -> None:
    """
    Try to enqueue a Celery task.
    If Celery/Redis is unavailable, fall back to FastAPI BackgroundTasks.
    """
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
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
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
                    },
                },
                f,
            )
        logger.error(f"Inline extraction failed for job {job_id}: {exc}", exc_info=True)


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
    logger.info(
        f"Upload complete | job={job_id} | file={file.filename} | size={size_bytes}"
    )

    _enqueue_or_run(job_id, background_tasks)

    return UploadResponse(
        job_id=job_id,
        filename=file.filename or "unknown.pdf",
        size_bytes=size_bytes,
    )
