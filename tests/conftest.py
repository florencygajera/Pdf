"""
Pytest configuration and shared fixtures.
"""

import os
import sys

# Ensure app module is importable from test runner
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set test environment before any app imports
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("UPLOAD_DIR", "/tmp/pdf_test_uploads")
os.environ.setdefault("OUTPUT_DIR", "/tmp/pdf_test_outputs")
