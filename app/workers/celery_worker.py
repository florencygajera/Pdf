"""
Celery Worker
Handles async/background PDF extraction jobs.

FIXES:
  C2 — self.retry() is only called for genuinely transient errors.
  C3 — RuntimeError removed from _RETRYABLE_ERRORS (too broad — many
        application-level errors raise RuntimeError, e.g. PaddleOCR import
        failures, shape mismatches, config errors; these should be permanent
        failures, not retried).
  C4 — IOError removed from _RETRYABLE_ERRORS; IOError is an alias for OSError
        since Python 3.3, so listing both was redundant.

After fixes:  _RETRYABLE_ERRORS = (OSError, MemoryError)
  - OSError covers all filesystem/network transient failures
  - MemoryError covers OOM scenarios worth retrying once
"""

import json
import os
import sys
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
    task_soft_time_limit=max(3, settings.CELERY_TASK_TIMEOUT - 5),
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    result_expires=settings.RESULT_EXPIRES_SECONDS,
    worker_concurrency=settings.effective_celery_worker_concurrency,
    worker_pool=settings.CELERY_WORKER_POOL,
)

celery_app.conf.broker_connection_retry_on_startup = True

if sys.platform == "win32":
    celery_app.conf.update(
        worker_pool="solo",
        worker_concurrency=1,
    )

task_logger = get_task_logger(__name__)

# FIX C3+C4: Only retry on filesystem / memory errors.
#   - Removed RuntimeError: too broad; catches application-level failures that
#     should be permanent (OCR import errors, config errors, shape mismatches).
#   - Removed IOError: it is an alias for OSError since Python 3.3, redundant.
_RETRYABLE_ERRORS = (OSError, MemoryError)


@celery_app.task(
    bind=True,
    name="extract_pdf",
    max_retries=2,
    default_retry_delay=10,
    throws=(ValueError,),
)
def extract_pdf_task(self, job_id: str) -> dict:
    """
    Background Celery task: run the full extraction pipeline for a given job.
    """
    pdf_path = get_upload_path(job_id)
    output_path = get_output_path(job_id)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Upload not found for job {job_id}: {pdf_path}")

    task_logger.info(f"Starting extraction task | job={job_id}")

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
        # Non-retryable: bad PDF, invalid content, validation failure
        task_logger.error(
            f"Validation error for job {job_id}: {exc}\n{traceback.format_exc()}"
        )
        _write_failed_result(job_id, output_path, str(exc))
        raise

    except _RETRYABLE_ERRORS as exc:
        # FIX C3: Only retry OSError / MemoryError — genuine transient failures
        task_logger.warning(f"Transient error for job {job_id} (will retry): {exc}")
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            _write_failed_result(job_id, output_path, f"Max retries exceeded: {exc}")
            return {
                "status": STATE_FAILED,
                "job_id": job_id,
                "error": str(exc),
            }

    except Exception as exc:
        # All other errors (including RuntimeError) are permanent failures — no retry.
        task_logger.error(
            f"Permanent error for job {job_id}: {exc}\n{traceback.format_exc()}"
        )
        _write_failed_result(job_id, output_path, str(exc))
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
        "expires_at": expires_at.isoformat(),
    }
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
