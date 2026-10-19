"""Pydantic schemas for text generation requests and responses."""

from __future__ import annotations

import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class GenerationRequest(BaseModel):
    """Payload for requesting text generation from a chosen model."""

    model_id: str = Field(
        ...,
        description="Target model identifier: 'mini-llm-25k' or 'mini-llm-100k'",
        examples=["mini-llm-100k"],
    )
    prompt: str = Field(
        ...,
        description="Initial seed text. Must contain at least 5 words.",
        examples=["The little mouse looked around the old house"],
    )
    max_tokens: int = Field(
        default=200,
        ge=1,
        le=5000,
        description="Maximum number of tokens to generate (must be <= 5000).",
        examples=[150],
    )
    max_characters: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional maximum character count limit for the generated continuation.",
        examples=[500],
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower is more deterministic, higher is more diverse.",
        examples=[0.8],
    )
    top_k: Optional[int] = Field(
        default=20,
        ge=1,
        description="Top-k sampling parameter. Retains only top k scoring tokens per step.",
        examples=[20],
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter cumulative probability threshold.",
        examples=[0.9],
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional random seed for reproducible generations.",
        examples=[42],
    )
    stop_at_eos: bool = Field(
        default=True,
        description="Whether to halt generation when the end-of-sequence token is predicted.",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_word_count(cls, value: str) -> str:
        """Ensure the prompt is non-empty and contains at least 5 words."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty or whitespace only.")

        # Split on whitespace to count words
        words = cleaned.split()
        if len(words) < 5:
            raise ValueError(
                f"Prompt must contain at least 5 words (got {len(words)} word{'s' if len(words) != 1 else ''}). "
                "Please provide a starter sentence or phrase with 5 to 7 words or more."
            )
        return cleaned


class GenerationResponse(BaseModel):
    """Response containing the generated continuation and operational metadata."""

    model_id: str = Field(..., description="The model ID that processed the request")
    prompt: str = Field(..., description="The original normalized prompt")
    completion: str = Field(..., description="The newly generated continuation text")
    full_text: str = Field(..., description="Prompt concatenated with the completion")
    prompt_tokens: int = Field(..., description="Number of tokens (bytes) in the input prompt")
    completion_tokens: int = Field(..., description="Number of tokens generated in the continuation")
    total_tokens: int = Field(..., description="Total tokens (prompt + completion)")
    character_count: int = Field(..., description="Character length of the generated continuation")
    truncated: bool = Field(..., description="True if generation stopped due to requested token/char limits")
    inference_duration_ms: float = Field(..., description="Inference latency in milliseconds")
