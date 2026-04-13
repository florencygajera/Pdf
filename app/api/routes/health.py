"""
Health Check Routes
/ping      — unauthenticated liveness stub for Docker / Kubernetes probes
/healthz   — liveness probe (requires API key when one is configured)
/readyz    — readiness probe (checks Redis/Celery connectivity)

FIX: Added /ping — a zero-auth, zero-dependency endpoint so Docker HEALTHCHECK
     and k8s liveness probes work without embedding the API key in the manifest.
     /healthz still enforces API key authentication as before.
"""

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.config.settings import settings
from app.api.security import require_api_key
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_START_TIME = time.time()


# ── Unauthenticated liveness stub ────────────────────────────────────────────
# FIX: Docker HEALTHCHECK and k8s liveness probes cannot supply an API key.
# /ping is intentionally minimal — it only checks that the process is alive
# and the event loop is running. No auth, no Redis, no heavy logic.
@router.get("/ping", include_in_schema=False)
async def ping():
    """Unauthenticated liveness stub for infrastructure probes."""
    return {"pong": True}


@router.get(
    "/healthz",
    summary="Liveness probe",
    dependencies=[Depends(require_api_key)],
)
async def health_live():
    """
    Kubernetes liveness probe — returns 200 if process is alive.
    Requires API key when one is configured.
    For internal cluster probes, configure your probe to include X-API-Key header,
    or use /ping which requires no authentication.
    """
    return {"status": "ok", "uptime_seconds": round(time.time() - _START_TIME, 1)}


@router.get(
    "/readyz",
    summary="Readiness probe",
    dependencies=[Depends(require_api_key)],
)
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

    # Check Celery worker heartbeat
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
