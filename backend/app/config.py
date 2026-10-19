"""Application configuration managed via Pydantic Settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for Mini LLM Backend."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    APP_HOST: str = Field(default="0.0.0.0", description="Host to bind the server")
    APP_PORT: int = Field(default=8000, description="Port to bind the server")
    APP_ENV: str = Field(default="development", description="Application environment")

    # CORS origins
    CORS_ALLOWED_ORIGINS: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated allowed CORS origins",
    )

    # Model checkpoint paths
    MODEL_25K_CHECKPOINT: str = Field(
        default="../mini-llm-lab/artifacts/experiments/tiny_25k_baseline_b288_v1/checkpoints/best.pt",
        description="Path to 25k baseline model checkpoint",
    )
    MODEL_100K_CHECKPOINT: str = Field(
        default="../mini-llm-lab/artifacts/experiments/tiny_100k_b1280_v1/checkpoints/best.pt",
        description="Path to 100k model checkpoint",
    )

    # Device selection
    DEVICE: str = Field(default="auto", description="Device to use ('auto', 'cuda', 'cpu')")

    # Text Generation limits
    MAX_ALLOWED_TOKENS: int = Field(default=5000, description="Absolute maximum allowed tokens")
    DEFAULT_MAX_TOKENS: int = Field(default=200, description="Default max tokens to generate")
    DEFAULT_TEMPERATURE: float = Field(default=0.8, description="Default sampling temperature")
    DEFAULT_TOP_K: int = Field(default=20, description="Default top-k value")
    DEFAULT_TOP_P: float = Field(default=0.9, description="Default nucleus sampling top-p")

    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins as a stripped list."""
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

    def resolve_path(self, path_str: str) -> Path:
        """Resolve a relative or absolute path against the backend directory."""
        path = Path(path_str)
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        return path

    @property
    def model_25k_resolved_path(self) -> Path:
        """Absolute Path to the 25k model checkpoint."""
        return self.resolve_path(self.MODEL_25K_CHECKPOINT)

    @property
    def model_100k_resolved_path(self) -> Path:
        """Absolute Path to the 100k model checkpoint."""
        return self.resolve_path(self.MODEL_100K_CHECKPOINT)


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
