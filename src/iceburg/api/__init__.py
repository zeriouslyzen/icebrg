"""
ICEBURG API Module
FastAPI server for ICEBURG 2.0
"""

from .server import create_app, app

__all__ = [
    "create_app",
    "app",
]

