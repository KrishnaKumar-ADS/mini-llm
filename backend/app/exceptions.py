"""Application domain exceptions and HTTP error handlers."""

from __future__ import annotations

import logging
from typing import Any, Dict
from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("mini_llm.api")


class MiniLLMException(Exception):
    """Base class for domain exceptions."""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST, error_code: str = "BAD_REQUEST") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class ModelNotFoundError(MiniLLMException):
    """Raised when an unknown model ID is requested."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            message=f"Model '{model_id}' was not found. Valid models are 'mini-llm-25k' and 'mini-llm-100k'.",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="MODEL_NOT_FOUND",
        )


class ModelNotLoadedError(MiniLLMException):
    """Raised when a requested model is registered but failed to load."""

    def __init__(self, model_id: str, reason: str = "") -> None:
        super().__init__(
            message=f"Model '{model_id}' is unavailable. {reason}".strip(),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MODEL_NOT_LOADED",
        )


class PromptValidationError(MiniLLMException):
    """Raised when the prompt does not meet operational requirements."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="INVALID_PROMPT",
        )


class GenerationExecutionError(MiniLLMException):
    """Raised when an error occurs during model forward pass or generation."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"Generation failed: {message}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="GENERATION_ERROR",
        )


async def mini_llm_exception_handler(request: Request, exc: MiniLLMException) -> JSONResponse:
    """Serialize custom domain exceptions into clean JSON responses."""
    logger.warning("Domain exception on %s %s: %s", request.method, request.url.path, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "path": request.url.path,
        },
    )
