"""
PDF Extraction System - FastAPI Application Entry Point
Production-grade hybrid PDF extraction for government documents.

FIXES:
  - CORS allow_methods now includes DELETE (was missing — DELETE /extract/{id}
    was blocked for cross-origin requests from the frontend)
  - app.state.limiter now uses the shared Limiter from app.api.limiter (was
    creating a separate instance, causing two independent rate-limit stores)
  - Lifespan shutdown now calls shutdown_ocr_executor() to cleanly terminate
    the ProcessPoolExecutor for OCR workers (was leaking zombie processes)
  - OCR pool is pre-warmed at startup in non-testing environments to avoid a
    5-10 second cold-start penalty on the first OCR request
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.limiter import limiter  # FIX: single shared limiter instance
from app.api.routes import extract, health, upload
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("🚀 PDF Extraction Service starting up...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Max file size: {settings.MAX_FILE_SIZE_MB} MB")

    # Purge stale uploads left from a previous run
    try:
        from app.utils.file_handler import purge_stale_uploads

        purge_stale_uploads()
    except Exception as exc:
        logger.warning(f"Startup purge failed (non-fatal): {exc}")

    # FIX: Pre-warm PaddleOCR pool in background so the first real request
    # doesn't pay the 5-10s model-load penalty. Skipped in test environments
    # where PaddleOCR is not installed/needed.
    if not settings.is_testing:
        try:
            import asyncio
            import concurrent.futures

            loop = asyncio.get_event_loop()

            def _warm_ocr():
                try:
                    from app.services.ocr_extractor import _get_ocr_pool

                    pool = _get_ocr_pool()
                    # Borrow and immediately return — triggers model load
                    with pool.borrow():
                        pass
                    logger.info("OCR pool pre-warmed successfully.")
                except Exception as exc:
                    logger.debug(f"OCR pre-warm skipped (non-fatal): {exc}")

            loop.run_in_executor(None, _warm_ocr)
        except Exception as exc:
            logger.debug(f"OCR pre-warm setup failed (non-fatal): {exc}")

    yield

    logger.info("🛑 PDF Extraction Service shutting down...")

    # FIX: Cleanly terminate the OCR ProcessPoolExecutor to avoid zombie
    # worker processes lingering after the main process exits.
    try:
        from app.services.ocr_extractor import shutdown_ocr_executor

        shutdown_ocr_executor()
        logger.info("OCR executor shut down cleanly.")
    except Exception as exc:
        logger.warning(f"OCR executor shutdown error (non-fatal): {exc}")


# ─── App Factory ─────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="Hybrid PDF Extraction System",
        description=(
            "Production-grade PDF extractor supporting digital + scanned PDFs. "
            "Handles government documents, tables, multi-language OCR, and large files."
        ),
        version="1.0.0",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    if FRONTEND_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

    # ── Rate Limiting — shared instance ───────────────────────────────────
    # FIX: use the shared limiter (app.api.limiter) so all route modules
    # share the same bucket store. Previously main.py created its own
    # Limiter() which was only used for exception handling, while route
    # modules each had separate Limiter() instances for enforcement.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── CORS ──────────────────────────────────────────────────────────────
    # FIX: Added "DELETE" to allow_methods. The cancel endpoint
    # DELETE /api/v1/extract/{job_id} was previously blocked for any
    # cross-origin request from the frontend.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],  # FIX: added DELETE
        allow_headers=["*"],
    )

    # ── Trusted Host ──────────────────────────────────────────────────────
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )

    # ── Request Timing Middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = round(time.time() - start, 4)
        response.headers["X-Process-Time"] = str(elapsed)
        return response

    @app.get("/", include_in_schema=False)
    async def homepage():
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse(
            status_code=404,
            content={"detail": "Frontend not available."},
        )

    # ── Global Exception Handler ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc)
                if settings.ENVIRONMENT != "production"
                else "Contact support.",
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router, tags=["Health"])
    app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
    app.include_router(extract.router, prefix="/api/v1", tags=["Extract"])

    return app


app = create_app()
