"""
PDF Extraction System - FastAPI Application Entry Point
Production-grade hybrid PDF extraction for government documents.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import extract, health, upload
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

# ─── Rate Limiter ────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("🚀 PDF Extraction Service starting up...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Max file size: {settings.MAX_FILE_SIZE_MB} MB")
    yield
    logger.info("🛑 PDF Extraction Service shutting down...")


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

    # ── Rate Limiting ──────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── CORS ───────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Trusted Host ───────────────────────────────────────────────────────
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )

    # ── Request Timing Middleware ──────────────────────────────────────────
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

    # ── Global Exception Handler ───────────────────────────────────────────
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

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(health.router, tags=["Health"])
    app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
    app.include_router(extract.router, prefix="/api/v1", tags=["Extract"])

    return app


app = create_app()
