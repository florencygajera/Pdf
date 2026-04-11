"""
Extract Routes
GET  /api/v1/extract/{job_id}         — poll status / get final result
GET  /api/v1/extract/{job_id}/text    — get only the clean text
GET  /api/v1/extract/{job_id}/tables  — get only tables as JSON
DELETE /api/v1/extract/{job_id}       — cancel and clean up job files
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config.constants import (
    RESULT_NOT_FOUND,
    RESULT_EXPIRED,
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
from app.utils.file_handler import (
    cleanup_job_files,
    get_output_path,
    get_upload_path,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["Extract"])
limiter = Limiter(key_func=get_remote_address)


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
    status, pct, msg, data = await _resolve_job_status(job_id)

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
async def get_job_status(job_id: str):
    """
    Lightweight status check (no full result body).
    Returns just status + progress percentage.
    """
    status, pct, msg, _ = await _resolve_job_status(job_id)
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
    status, _, _, data = await _resolve_job_status(job_id)
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
async def get_tables_only(job_id: str):
    """Return only extracted tables for a completed job."""
    status, _, _, data = await _resolve_job_status(job_id)
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
async def delete_job(job_id: str):
    """
    Cancel a running job (best-effort) and delete all associated files.
    """
    asyncio.create_task(_revoke_celery_job(job_id))

    cleanup_job_files(job_id)
    return {"job_id": job_id, "message": "Job cancelled and files cleaned up."}


# ── Helpers ─────────────────────────────────────────────────────────────────────


async def _load_output(job_id: str) -> dict:
    """Load the JSON result file for a completed job (async, non-blocking)."""
    output_path = get_output_path(job_id)
    if not output_path.exists():
        return {"_state": RESULT_NOT_FOUND}

    try:
        # Use asyncio.to_thread to run blocking sync file I/O in thread pool
        # This avoids blocking the event loop during disk reads
        content = await asyncio.to_thread(output_path.read_text, encoding="utf-8")
        data = json.loads(content)
    except (OSError, json.JSONDecodeError):
        return {"_state": RESULT_NOT_FOUND}

    expires_at = data.get("expires_at")
    if expires_at:
        try:
            expiry_dt = datetime.fromisoformat(expires_at)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
            if expiry_dt <= datetime.now(timezone.utc):
                return {"_state": RESULT_EXPIRED, "expires_at": expires_at}
        except (ValueError, TypeError):
            pass

    return data


async def _resolve_job_status(job_id: str):
    """
    Resolve current job status, handling all three cases:
      - unknown job_id (None)
      - known but still processing (202)
      - done, expired, or failed (result payload)
    """
    output_path = get_output_path(job_id)
    upload_path = get_upload_path(job_id)

    # Fast path: output exists → load and return it
    if output_path.exists():
        data = await _load_output(job_id)
        state_marker = data.get("_state")

        if state_marker == RESULT_EXPIRED:
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

    # Upload exists but no output → still processing
    if upload_path.exists():
        return STATE_PROCESSING, 5.0, "Processing started.", None

    return None, 0.0, "", None


def _get_celery_job_state(job_id: str):
    """Check Celery AsyncResult state (if Celery is configured)."""
    try:
        from celery.result import AsyncResult
    except Exception:
        return None, 0.0, ""

    try:
        result = AsyncResult(job_id)
        state = result.state
        meta = result.info or {}

        if state == "PENDING":
            return STATE_PENDING, 0.0, "Waiting for worker."
        elif state == "STARTED":
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


async def _revoke_celery_job(job_id: str) -> None:
    """Revoke a Celery task by ID (best-effort)."""
    try:
        from celery.result import AsyncResult
        from celery.task.control import revoke
    except Exception:
        return

    try:
        revoke(job_id, terminate=False)
    except Exception as exc:
        logger.debug(f"Celery revocation failed: {exc}")
