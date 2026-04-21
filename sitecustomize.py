"""
Early process bootstrap for local test and runtime stability.

Pytest may resolve temporary directories before conftest.py is imported, so we
set workspace-local temp paths here as early as possible.
"""

import os
import inspect
import tempfile
from pathlib import Path

_workspace_temp = Path(__file__).resolve().parent / "_tmp"
_workspace_temp.mkdir(parents=True, exist_ok=True)

_pytest_root = _workspace_temp / f"pytest-of-{os.environ.get('USERNAME', 'user')}"
_pytest_root.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TMPDIR", str(_workspace_temp))
os.environ.setdefault("TEMP", str(_workspace_temp))
os.environ.setdefault("TMP", str(_workspace_temp))
tempfile.tempdir = str(_workspace_temp)

try:
    from starlette.routing import Router as _StarletteRouter

    _router_init_sig = inspect.signature(_StarletteRouter.__init__)
    if "on_startup" not in _router_init_sig.parameters:
        _router_init = _StarletteRouter.__init__

        def _compat_router_init(
            self,
            *args,
            on_startup=None,
            on_shutdown=None,
            lifespan=None,
            **kwargs,
        ):
            return _router_init(self, *args, **kwargs)

        _StarletteRouter.__init__ = _compat_router_init
except Exception:
    pass
