# server/app.py
# Required entry point for OpenEnv multi-mode deployment validator.
# This simply re-exports the main FastAPI app from the root app.py.

import sys
import os

# Ensure root is on path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401 — re-export for OpenEnv validator

__all__ = ["app"]