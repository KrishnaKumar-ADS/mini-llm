"""Pydantic schemas for model information and listing."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Metadata describing a loaded Mini LLM model."""

    model_id: str = Field(..., description="Unique model identifier, e.g. 'mini-llm-25k'")
    display_name: str = Field(..., description="Human-readable model name")
    description: str = Field(..., description="Overview of model training and capacity")
    training_subset: str = Field(..., description="Training dataset subset size ('25k' or '100k')")
    parameter_count: int = Field(..., description="Total trainable model parameters")
    context_length: int = Field(..., description="Maximum context window block size in tokens")
    validation_loss: Optional[float] = Field(None, description="Best recorded validation loss")
    validation_perplexity: Optional[float] = Field(None, description="Best recorded validation perplexity")
    batch_size: Optional[int] = Field(None, description="Training batch size")
    training_steps: int = Field(default=10000, description="Completed training steps")


class ModelListResponse(BaseModel):
    """Response schema containing all available models."""

    models: List[ModelInfo] = Field(..., description="List of registered models")
