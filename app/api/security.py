"""Optional API-key authentication helpers."""

from fastapi import HTTPException, Request, status

from app.config.settings import settings


def require_api_key(request: Request) -> None:
    """Enforce an API key for protected routes."""
    expected = getattr(settings, "API_KEY", None)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured on the server.",
        )

    provided = request.headers.get("X-API-Key")
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
