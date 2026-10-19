"""Routers package."""

from backend.app.routers.generation import router as generation_router
from backend.app.routers.health import router as health_router
from backend.app.routers.models import router as models_router

__all__ = ["generation_router", "health_router", "models_router"]
