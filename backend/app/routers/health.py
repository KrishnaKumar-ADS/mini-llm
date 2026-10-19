"""Health check router reporting operational status and hardware capabilities."""

from __future__ import annotations

from typing import Any, Dict
import torch
from fastapi import APIRouter

from backend.app.services.model_registry import ModelRegistry

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Service Health and Diagnostics")
async def get_health_status() -> Dict[str, Any]:
    """Return runtime health, device diagnostics, and loaded models."""
    registry = ModelRegistry.get_instance()
    is_cuda = torch.cuda.is_available()

    return {
        "status": "healthy" if registry.is_healthy() else "initializing",
        "device": str(registry.device),
        "cuda_available": is_cuda,
        "cuda_device_name": torch.cuda.get_device_name(0) if is_cuda else None,
        "loaded_models": registry.loaded_model_ids,
    }
