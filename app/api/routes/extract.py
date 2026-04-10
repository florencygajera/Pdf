"""
Extract Routes
GET  /api/v1/extract/{job_id}         — poll status / get final result
GET  /api/v1/extract/{job_id}/text    — get only the clean text
GET  /api/v1/extract/{job_id}/tables  — get only tables as JSON
DELETE /api/v1/extract/{job_id}       — cancel and clean up job files
"""

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config.constants import (
    STATE_DONE,
    STATE_FAILED,
    STATE_PENDING,
    STATE_PROCESSING,
)
from app.config.settings import settings
from app.models.response_model import (
    ErrorResponse,
    ExtractionResult,
    JobStatus,
    TableData,
)
from app.utils.file_handler import cleanup_job_files, get_output_path, get_upload_path
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)


def _get_celery_job_state(job_id: str):
    """
    Query Celery backend for task state.
    Returns (state_str, progress_pct, message) or None if Celery unavailable.
    """
    try:
        from celery.result import AsyncResult
        from app.workers.celery_worker import celery_app

        result = AsyncResult(job_id, app=celery_app)
        state = result.state

        if state == "PENDING":
            return STATE_PENDING, 0.0, "Queued, waiting for worker."
        elif state == STATE_PROCESSING:
            meta = result.info or {}
            return STATE_PROCESSING, meta.get("progress", 0.0), meta.get("message", "")
        elif state == "SUCCESS":
            return STATE_DONE, 100.0, "Complete."
        elif state == "FAILURE":
            return STATE_FAILED, 0.0, str(result.info)
        else:
            return state.lower(), 0.0, ""

    except Exception as exc:
        logger.debug(f"Celery state check failed: {exc}")
        return None, 0.0, ""


def _load_output(job_id: str) -> dict:
    """Load the JSON result file for a completed job."""
    output_path = get_output_path(job_id)
    if not output_path.exists():
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_job_status(job_id: str):
    """
    Determine job status from multiple sources:
    1. Celery backend (if available)
    2. Output file on disk (fastest check)
    3. Upload file existence (pending)

    Returns (status, progress, message, result_data_or_None)
    """
    output_path = get_output_path(job_id)
    upload_path = get_upload_path(job_id)

    # Fast path: output file exists → job is done or failed
    if output_path.exists():
        data = _load_output(job_id)
        status = data.get("status", STATE_DONE)
        return status, 100.0, "", data

    # Check Celery
    celery_status, pct, msg = _get_celery_job_state(job_id)
    if celery_status:
        return celery_status, pct, msg, None

    # Upload exists but no output → still processing
    if upload_path.exists():
        return STATE_PROCESSING, 5.0, "Processing started.", None

    # Neither exists → unknown job_id
    return None, 0.0, "", None


@router.get(
    "/extract/{job_id}",
    response_model=ExtractionResult,
    responses={
        202: {"description": "Job still processing"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Get extraction result",
)
@limiter.limit(settings.RATE_LIMIT_EXTRACT)
async def get_extraction_result(request: Request, job_id: str):
    """
    Poll for extraction results.

    - **202** → still processing (check `status` and `progress_percent`)
    - **200** → complete (full result in body)
    - **404** → unknown job_id
    """
    status, pct, msg, data = _resolve_job_status(job_id)

    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. Upload a PDF first.",
        )

    if status == STATE_FAILED:
        raise HTTPException(
            status_code=500,
            detail=data.get("error", "Extraction failed.") if data else msg,
        )

    if status in (STATE_PROCESSING, STATE_PENDING):
        # Return 202 with progress info
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": status,
                "progress_percent": pct,
                "message": msg or "Processing in progress. Poll again shortly.",
            },
        )

    # Status = done — return full result
    if not data:
        raise HTTPException(
            status_code=500, detail="Output file missing despite done status."
        )

    try:
        return ExtractionResult(**data)
    except Exception as exc:
        logger.error(f"Failed to parse result for job {job_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Result parsing error: {exc}")


@router.get(
    "/extract/{job_id}/status",
    response_model=JobStatus,
    summary="Lightweight status check",
)
async def get_job_status(job_id: str):
    """
    Lightweight status check (no full result body).
    Returns just status + progress percentage.
    """
    status, pct, msg, _ = _resolve_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JobStatus(
        job_id=job_id,
        status=status,
        progress_percent=pct,
        message=msg,
    )


@router.get(
    "/extract/{job_id}/text",
    summary="Get only extracted text",
)
async def get_text_only(job_id: str):
    """Return only the clean extracted text for a completed job."""
    status, _, _, data = _resolve_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if status != STATE_DONE:
        raise HTTPException(status_code=202, detail=f"Job status: {status}")
    return {"job_id": job_id, "text": data.get("text", "")}


@router.get(
    "/extract/{job_id}/tables",
    response_model=List[TableData],
    summary="Get only extracted tables",
)
async def get_tables_only(job_id: str):
    """Return only extracted tables for a completed job."""
    status, _, _, data = _resolve_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if status != STATE_DONE:
        raise HTTPException(status_code=202, detail=f"Job status: {status}")
    tables = data.get("tables", [])
    return [TableData(**t) for t in tables]


@router.delete(
    "/extract/{job_id}",
    summary="Cancel and clean up job",
)
async def delete_job(job_id: str):
    """
    Cancel a running job (best-effort) and delete all associated files.
    """
    # Best-effort Celery revoke
    try:
        from app.workers.celery_worker import celery_app

        celery_app.control.revoke(job_id, terminate=True, signal="SIGKILL")
        logger.info(f"Celery task {job_id} revoked.")
    except Exception:
        pass

    cleanup_job_files(job_id)
    return {"job_id": job_id, "message": "Job cancelled and files cleaned up."}
