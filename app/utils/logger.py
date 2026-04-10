"""
Structured logger — outputs JSON in production, readable text in development.
"""

import logging
import sys
from typing import Optional

try:
    import structlog

    _USE_STRUCTLOG = True
except ImportError:
    _USE_STRUCTLOG = False

from app.config.settings import settings


def _configure_stdlib_logger(name: str) -> logging.Logger:
    """Fallback: plain Python logger with a readable formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger instance.
    Uses structlog if available (JSON output), else stdlib.
    """
    if _USE_STRUCTLOG and settings.LOG_FORMAT == "json":
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        return structlog.get_logger(name)

    return _configure_stdlib_logger(name or "pdf_extractor")
