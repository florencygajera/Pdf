"""
Health Check Routes
/healthz   — liveness probe (always returns 200 if app is running)
/readyz    — readiness probe (checks Redis/Celery connectivity)
"""

import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.config.settings import settings
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_START_TIME = time.time()


@router.get("/healthz", summary="Liveness probe")
async def health_live():
    """Kubernetes liveness probe — returns 200 if process is alive."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _START_TIME, 1)}


@router.get("/readyz", summary="Readiness probe")
async def health_ready():
    """
    Kubernetes readiness probe.
    Checks Redis connectivity and confirms at least one Celery worker responds.
    """
    checks = {"app": "ok"}
    require_worker = (
        settings.READINESS_REQUIRE_WORKER
        or settings.ENVIRONMENT.lower() == "production"
    )

    # Check Redis
    try:
        import redis

        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        r.ping()
        checks["redis"] = "ok"
    except ImportError:
        checks["redis"] = "redis package not installed"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"
        logger.warning(f"Redis readiness check failed: {exc}")

    # Check Celery worker heartbeat so readiness reflects actual processing capacity.
    try:
        from app.workers.celery_worker import celery_app

        inspector = celery_app.control.inspect(timeout=1.5)
        ping_result = inspector.ping() if inspector else None
        checks["celery_worker"] = "ok" if ping_result else "no workers responding"
    except Exception as exc:
        checks["celery_worker"] = f"error: {exc}"
        logger.warning(f"Celery readiness check failed: {exc}")

    all_ok = checks.get("app") == "ok" and checks.get("redis") == "ok"
    if require_worker:
        all_ok = all_ok and checks.get("celery_worker") == "ok"

    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={
            "status": "ready" if all_ok else "not_ready",
            "checks": checks,
        },
        status_code=status_code,
    )
