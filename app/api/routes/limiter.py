"""
Shared SlowAPI rate-limiter instance.

FIX: Previously each route file (upload.py, extract.py, main.py) created its
own Limiter() object. Because each instance maintains its own in-memory bucket
store, rate-limit counters were NOT shared — a client could exceed the limit on
one endpoint without it counting against another. All route modules must import
this single instance so buckets are unified.

CRITICAL FIX: This file was missing entirely — the limiter was defined at
app/api/routes/limiter.py but imported as `from app.api.limiter import limiter`
everywhere (main.py, upload.py, extract.py). This caused an ImportError crash
on startup, preventing the app from running at all.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
