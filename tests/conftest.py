"""
Pytest configuration and shared fixtures.
"""

import os
import sys
import tempfile
from uuid import uuid4
from pathlib import Path

import pytest

# Ensure app module is importable from test runner
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set test environment before any app imports
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("API_KEY", "test-api-key")

# Keep pytest and app temp files inside the workspace on Windows machines where
# the default user temp directory may be inaccessible in sandboxed environments.
_workspace_temp = Path(__file__).resolve().parents[1] / "_tmp"
_workspace_temp.mkdir(parents=True, exist_ok=True)
(_workspace_temp / f"pytest-of-{os.environ.get('USERNAME', 'user')}").mkdir(
    parents=True, exist_ok=True
)
os.environ.setdefault("TMPDIR", str(_workspace_temp))
os.environ.setdefault("TEMP", str(_workspace_temp))
os.environ.setdefault("TMP", str(_workspace_temp))
tempfile.tempdir = str(_workspace_temp)

os.environ.setdefault("UPLOAD_DIR", str(_workspace_temp / "uploads"))
os.environ.setdefault("OUTPUT_DIR", str(_workspace_temp / "outputs"))


@pytest.fixture
def tmp_path():
    """
    Workspace-local replacement for pytest's tmp_path fixture.

    The default temp factory resolves to a user-profile location on this
    machine, which is not writable in the current sandboxed environment.
    """
    path = _workspace_temp / "pytest-fixtures" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path
