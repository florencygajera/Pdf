"""
Health Check Routes
/healthz   — liveness probe (always returns 200 if app is running)
/readyz    — readiness probe (checks Redis/Celery connectivity)
"""

import time

from fastapi import APIRouter
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
    Checks Redis connectivity (required for Celery task queue).
    """
    checks = {"app": "ok"}

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

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={
            "status": "ready" if all_ok else "not_ready",
            "checks": checks,
        },
        status_code=status_code,
    )
