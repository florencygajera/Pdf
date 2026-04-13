"""
Extract Routes
GET  /api/v1/extract/{job_id}         — poll status / get final result
GET  /api/v1/extract/{job_id}/status  — lightweight progress check
GET  /api/v1/extract/{job_id}/text    — get only the clean text
GET  /api/v1/extract/{job_id}/tables  — get only tables as JSON
DELETE /api/v1/extract/{job_id}       — cancel and clean up job files

FIX: Imports the shared Limiter from app.api.limiter (was creating a new
     instance locally — rate-limit buckets were NOT shared with upload.py).
FIX: get_job_status, get_text_only, get_tables_only, and delete_job now
     accept `request: Request` and carry @limiter.limit() decorators — these
     sub-endpoints were previously completely unthrottled.
"""

import json
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.limiter import limiter  # FIX: shared limiter
from app.config.constants import (
    STATE_DONE,
    STATE_FAILED,
    STATE_PENDING,
    STATE_PROCESSING,
)
from app.config.settings import settings
from app.api.security import require_api_key
from app.models.response_model import (
    ErrorResponse,
    ExtractionResult,
    JobStatus,
    TableData,
)
from app.utils.file_handler import cleanup_job_files, get_output_path, get_upload_path
from app.utils.logger import get_logger

router = APIRouter(dependencies=[Depends(require_api_key)])
logger = get_logger(__name__)

RESULT_NOT_FOUND = "not_found"
RESULT_EXPIRED = "expired"


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
        return {"_state": RESULT_NOT_FOUND}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"_state": RESULT_NOT_FOUND}

    expires_at = data.get("expires_at")
    if expires_at:
        try:
            expiry_dt = datetime.fromisoformat(expires_at)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
            if expiry_dt <= datetime.now(timezone.utc):
                output_path.unlink(missing_ok=True)
                return {"_state": RESULT_EXPIRED}
        except Exception:
            pass
    else:
        try:
            age_seconds = (
                datetime.now(timezone.utc).timestamp() - output_path.stat().st_mtime
            )
            if age_seconds > settings.RESULT_EXPIRES_SECONDS:
                output_path.unlink(missing_ok=True)
                return {"_state": RESULT_EXPIRED}
        except Exception:
            pass

    return data


def _resolve_job_status(job_id: str):
    """
    Determine job status from multiple sources:
    1. Output file on disk (fastest — avoids Celery round-trip for done jobs)
    2. Celery backend (for in-flight jobs)
    3. Upload file existence (fallback pending signal)

    Returns (status, progress, message, result_data_or_None)
    """
    output_path = get_output_path(job_id)
    upload_path = get_upload_path(job_id)

    # Fast path: output file exists → job is done or failed
    if output_path.exists():
        data = _load_output(job_id)
        state_marker = data.get("_state")
        if state_marker == RESULT_EXPIRED:
            cleanup_job_files(job_id)
            return RESULT_EXPIRED, 0.0, "Result expired.", None
        if state_marker == RESULT_NOT_FOUND:
            cleanup_job_files(job_id)
            return None, 0.0, "", None
        status = data.get("status", STATE_DONE)
        return status, 100.0, "", data

    # If we never saw an upload, this job ID is unknown.
    if not upload_path.exists():
        return None, 0.0, "", None

    # Check Celery only for known uploads.
    celery_status, pct, msg = _get_celery_job_state(job_id)
    if celery_status:
        return celery_status, pct, msg, None

    # Upload exists but no output and no Celery state → still processing
    return STATE_PROCESSING, 5.0, "Processing started.", None


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
    - **410** → result expired
    """
    status, pct, msg, data = _resolve_job_status(job_id)

    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. Upload a PDF first.",
        )

    if status == RESULT_EXPIRED:
        raise HTTPException(
            status_code=410,
            detail=f"Job '{job_id}' has expired. Please upload the PDF again.",
        )

    if status == STATE_FAILED:
        raise HTTPException(
            status_code=500,
            detail=data.get("error", "Extraction failed.") if data else msg,
        )

    if status in (STATE_PROCESSING, STATE_PENDING):
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
@limiter.limit(settings.RATE_LIMIT_EXTRACT)  # FIX: was unthrottled
async def get_job_status(request: Request, job_id: str):  # FIX: added request: Request
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
@limiter.limit(settings.RATE_LIMIT_EXTRACT)  # FIX: was unthrottled
async def get_text_only(request: Request, job_id: str):  # FIX: added request: Request
    """Return only the clean extracted text for a completed job."""
    status, _, _, data = _resolve_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if status == RESULT_EXPIRED:
        raise HTTPException(
            status_code=410,
            detail=f"Job '{job_id}' has expired. Please upload the PDF again.",
        )
    if status != STATE_DONE:
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": status,
                "message": "Still processing.",
            },
        )
    return {"job_id": job_id, "text": data.get("text", "")}


@router.get(
    "/extract/{job_id}/tables",
    response_model=List[TableData],
    summary="Get only extracted tables",
)
@limiter.limit(settings.RATE_LIMIT_EXTRACT)  # FIX: was unthrottled
async def get_tables_only(request: Request, job_id: str):  # FIX: added request: Request
    """Return only extracted tables for a completed job."""
    status, _, _, data = _resolve_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if status == RESULT_EXPIRED:
        raise HTTPException(
            status_code=410,
            detail=f"Job '{job_id}' has expired. Please upload the PDF again.",
        )
    if status != STATE_DONE:
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": status,
                "message": "Still processing.",
            },
        )
    tables = data.get("tables", [])
    return [TableData(**t) for t in tables]


@router.delete(
    "/extract/{job_id}",
    summary="Cancel and clean up job",
)
@limiter.limit(settings.RATE_LIMIT_EXTRACT)  # FIX: was unthrottled
async def delete_job(request: Request, job_id: str):  # FIX: added request: Request
    """
    Cancel a running job (best-effort) and delete all associated files.
    """
    # Best-effort Celery revoke
    try:
        from app.workers.celery_worker import celery_app

        celery_app.control.revoke(job_id, terminate=True, signal="SIGTERM")
        import asyncio

        await asyncio.sleep(2)
        logger.info(f"Celery task {job_id} revoked.")
    except Exception:
        pass

    cleanup_job_files(job_id)
    return {"job_id": job_id, "message": "Job cancelled and files cleaned up."}
