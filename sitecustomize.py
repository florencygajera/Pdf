"""
Early process bootstrap for local test and runtime stability.

Pytest may resolve temporary directories before conftest.py is imported, so we
set workspace-local temp paths here as early as possible.
"""

import os
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
