"""Main FastAPI application entry point with lifespan management and CORS."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure mini-llm-lab/src is available on Python path if running standalone
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
SRC_DIR = PROJECT_ROOT / "mini-llm-lab" / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure backend directory is also on sys.path for direct imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.exceptions import MiniLLMException, mini_llm_exception_handler
from backend.app.routers.generation import router as generation_router
from backend.app.routers.health import router as health_router
from backend.app.routers.models import router as models_router
from backend.app.services.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("mini_llm.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager to pre-load models and handle shutdown."""
    settings = get_settings()
    logger.info("Starting Mini LLM API server (env=%s)...", settings.APP_ENV)

    # Initialize and pre-load all checkpoints into registry
    try:
        registry = ModelRegistry.get_instance()
        registry.load_models()
        logger.info("Successfully loaded models on device: %s", registry.device)
    except Exception as exc:
        logger.exception("Failed to initialize models during startup: %s", exc)
        raise

    yield

    logger.info("Mini LLM API server shutting down gracefully.")


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title="Mini LLM Inference API",
        description="Production-grade API for text completion using trained 25k and 100k Mini LLM models.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Setup Cross-Origin Resource Sharing (CORS)
    logger.info("Configuring CORS with allowed origins: %s", settings.cors_origins_list)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register custom domain exception handlers
    app.add_exception_handler(MiniLLMException, mini_llm_exception_handler)

    # Register routers under /api/v1 prefix
    api_v1_prefix = "/api/v1"
    app.include_router(health_router, prefix=api_v1_prefix)
    app.include_router(models_router, prefix=api_v1_prefix)
    app.include_router(generation_router, prefix=api_v1_prefix)

    # Also expose root health check for container/orchestrator probes
    app.include_router(health_router, prefix="")

    @app.get("/", tags=["Root"])
    async def root_status() -> JSONResponse:
        """Root API metadata and documentation link."""
        return JSONResponse(
            content={
                "name": "Mini LLM Inference Service",
                "version": "1.0.0",
                "status": "online",
                "docs": "/docs",
                "endpoints": {
                    "health": f"{api_v1_prefix}/health",
                    "models": f"{api_v1_prefix}/models",
                    "generate": f"{api_v1_prefix}/generate",
                },
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development",
    )
