"""Router for executing text completion inference."""

from __future__ import annotations

import logging
from fastapi import APIRouter, status

from backend.app.schemas.generation import GenerationRequest, GenerationResponse
from backend.app.services.text_generator import TextGeneratorService

logger = logging.getLogger("mini_llm.api.generation")

router = APIRouter(prefix="/generate", tags=["Generation"])


@router.post(
    "",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Text Completion",
    description="Generate autoregressive continuation from an input prompt (min 5 words) up to the specified token/character limit.",
)
async def generate_text(request: GenerationRequest) -> GenerationResponse:
    """Generate completion using the designated model."""
    logger.info(
        "Generation requested: model=%s, max_tokens=%d, prompt_words=%d",
        request.model_id,
        request.max_tokens,
        len(request.prompt.split()),
    )
    service = TextGeneratorService()
    response = service.generate(request)
    logger.info(
        "Generation completed: model=%s, generated_tokens=%d, truncated=%s in %.2fms",
        response.model_id,
        response.completion_tokens,
        response.truncated,
        response.inference_duration_ms,
    )
    return response
