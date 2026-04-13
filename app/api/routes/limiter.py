"""
Shared SlowAPI rate-limiter instance.

FIX: Previously each route file (upload.py, extract.py, main.py) created its
own Limiter() object. Because each instance maintains its own in-memory bucket
store, rate-limit counters were NOT shared — a client could exceed the limit on
one endpoint without it counting against another. All route modules must import
this single instance so buckets are unified.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
