"""
Celery Worker
Handles async/background PDF extraction jobs.
Tasks are queued when a PDF is uploaded and polled via /extract/{job_id}.

Run worker with:
  celery -A app.workers.celery_worker worker --loglevel=info --concurrency=2
"""

import json
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger

from app.config.constants import STATE_DONE, STATE_FAILED, STATE_PROCESSING
from app.config.settings import settings
from app.pipelines.extraction_pipeline import (
    run_extraction_pipeline,
    save_result_to_disk,
)
from app.utils.file_handler import get_output_path, get_upload_path

# ── Celery App ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "pdf_extractor",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIMEOUT,
    # Keep a sensible grace window, but never let the soft limit go non-positive.
    task_soft_time_limit=max(30, settings.CELERY_TASK_TIMEOUT - 30),
    worker_prefetch_multiplier=1,  # One task at a time per worker
    task_acks_late=True,  # Acknowledge only after completion (safer)
    result_expires=settings.RESULT_EXPIRES_SECONDS,
)

task_logger = get_task_logger(__name__)


@celery_app.task(
    bind=True,
    name="extract_pdf",
    max_retries=2,
    default_retry_delay=10,
    throws=(ValueError,),  # Don't retry validation errors
)
def extract_pdf_task(self, job_id: str) -> dict:
    """
    Background Celery task: run the full extraction pipeline for a given job.

    Args:
        job_id: UUID of the upload job.

    Returns:
        dict with status and output_path.
    """
    pdf_path = get_upload_path(job_id)
    output_path = get_output_path(job_id)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Upload not found for job {job_id}: {pdf_path}")

    task_logger.info(f"Starting extraction task | job={job_id}")

    # Update state to PROCESSING so API can report progress
    self.update_state(
        state=STATE_PROCESSING,
        meta={"progress": 0, "message": "Starting extraction..."},
    )

    def progress_cb(step, total, stage):
        pct = round((step / total) * 100, 1)
        self.update_state(
            state=STATE_PROCESSING,
            meta={
                "progress": pct,
                "message": f"Processing {stage} chunk {step}/{total}",
            },
        )

    try:
        result = run_extraction_pipeline(
            pdf_path, job_id, progress_callback=progress_cb
        )
        save_result_to_disk(result, output_path)

        task_logger.info(f"Task complete | job={job_id} | output={output_path}")
        return {
            "status": STATE_DONE,
            "job_id": job_id,
            "output_path": str(output_path),
        }

    except SoftTimeLimitExceeded as exc:
        task_logger.error(f"Soft time limit exceeded for job {job_id}: {exc}")
        _write_failed_result(job_id, output_path, "Task timed out.")
        raise

    except ValueError as exc:
        # Non-retryable: bad PDF, invalid content
        task_logger.error(f"Validation error for job {job_id}: {exc}")
        _write_failed_result(job_id, output_path, str(exc))
        raise

    except Exception as exc:
        task_logger.error(
            f"Unexpected error for job {job_id}: {exc}\n{traceback.format_exc()}"
        )
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            _write_failed_result(job_id, output_path, f"Max retries exceeded: {exc}")
            return {
                "status": STATE_FAILED,
                "job_id": job_id,
                "error": str(exc),
            }


def _write_failed_result(job_id: str, output_path: Path, error_msg: str) -> None:
    """Write a minimal failed-state JSON so the API can return an informative error."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=settings.RESULT_EXPIRES_SECONDS
    )
    data = {
        "job_id": job_id,
        "status": STATE_FAILED,
        "error": error_msg,
        "expires_at": expires_at.isoformat(),
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
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
