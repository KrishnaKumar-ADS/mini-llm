"""Router for discovering and inspecting available Mini LLM models."""

from __future__ import annotations

from fastapi import APIRouter, status

from backend.app.schemas.models import ModelInfo, ModelListResponse
from backend.app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Available Models",
    description="Retrieve all loaded Mini LLM models with training metadata and architecture specifications.",
)
async def list_models() -> ModelListResponse:
    """Return all active models from the model registry."""
    registry = ModelRegistry.get_instance()
    models = registry.list_models()
    return ModelListResponse(models=models)


@router.get(
    "/{model_id}",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    summary="Get Model Information",
    description="Retrieve detailed specifications and validation metrics for a specific model ID.",
)
async def get_model(model_id: str) -> ModelInfo:
    """Fetch metadata for a single model by identifier."""
    registry = ModelRegistry.get_instance()
    entry = registry.get_model_entry(model_id)
    return entry.metadata
