"""
Optional API-key authentication helpers.

FIX: When API_KEY is not configured, return 401 (not 500).
     In development/test environments with no key set, all requests are
     rejected with a clear message rather than an internal server error.
"""

from fastapi import HTTPException, Request, status

from app.config.settings import settings


def require_api_key(request: Request) -> None:
    """
    Enforce an API key for protected routes.

    FIX: Previously raised HTTP 500 when API_KEY was not configured on the server.
    Now raises HTTP 401 with a clear message so clients know what to do.
    In environments where API_KEY is intentionally empty (e.g. local dev),
    you can set API_KEY to an empty string to disable auth enforcement.
    """
    expected = getattr(settings, "API_KEY", None)

    # FIX: empty string means auth is disabled (dev convenience)

    if expected == "":
        return

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is not configured on the server. Set API_KEY in your .env file.",
        )

    provided = request.headers.get("X-API-Key")
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
