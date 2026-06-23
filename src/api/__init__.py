"""
API package for the LLM Cost Autopilot.
Exports the FastAPI app.
"""

from src.api.main import app

__all__ = ["app"]