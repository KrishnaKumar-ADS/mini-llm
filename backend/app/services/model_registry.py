"""Model registry for caching and serving Mini LLM checkpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model import MiniLLM
from mini_llm.tokenizer import ByteTokenizer

from backend.app.config import Settings, get_settings
from backend.app.exceptions import ModelNotFoundError, ModelNotLoadedError
from backend.app.schemas.models import ModelInfo

logger = logging.getLogger("mini_llm.registry")


@dataclass
class LoadedModelEntry:
    """Container for an instantiated in-memory model and its metadata."""

    model_id: str
    model: MiniLLM
    config: ModelConfig
    metadata: ModelInfo
    checkpoint_step: int
    best_loss: Optional[float]


class ModelRegistry:
    """Thread-safe singleton registry that loads and serves model checkpoints."""

    _instance: Optional[ModelRegistry] = None

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.tokenizer = ByteTokenizer()
        self.device = self._determine_device()
        self._models: Dict[str, LoadedModelEntry] = {}
        self._is_initialized = False

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        """Get or initialize the shared singleton registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _determine_device(self) -> torch.device:
        """Select torch device based on settings and available hardware."""
        req = self.settings.DEVICE.lower()
        if req == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("cuda")
        elif req == "cpu":
            return torch.device("cpu")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self) -> None:
        """Load both 25k and 100k checkpoints into memory."""
        if self._is_initialized:
            logger.info("ModelRegistry is already initialized.")
            return

        logger.info("Initializing ModelRegistry on device: %s", self.device)

        # Definitions for both models
        model_specs = [
            {
                "model_id": "mini-llm-25k",
                "display_name": "Mini LLM — 25K Baseline",
                "description": "Trained on a 25k stories subset with batch size 288 for 10,000 steps. Lightweight baseline model.",
                "path": self.settings.model_25k_resolved_path,
                "training_subset": "25k",
                "validation_loss": 0.7277,
                "validation_perplexity": 2.0703,
                "batch_size": 288,
                "steps": 10000,
            },
            {
                "model_id": "mini-llm-100k",
                "display_name": "Mini LLM — 100K Advanced",
                "description": "Trained on a 100k stories subset with batch size 1,280 for 10,000 steps on Tesla T4. Superior coherence and lower perplexity.",
                "path": self.settings.model_100k_resolved_path,
                "training_subset": "100k",
                "validation_loss": 0.6862,
                "validation_perplexity": 1.9862,
                "batch_size": 1280,
                "steps": 10000,
            },
        ]

        for spec in model_specs:
            self._load_single_model(spec)

        self._is_initialized = True
        logger.info(
            "ModelRegistry loaded %d model(s): %s",
            len(self._models),
            list(self._models.keys()),
        )

    def _load_single_model(self, spec: dict) -> None:
        """Load and cache an individual model checkpoint."""
        model_id = spec["model_id"]
        checkpoint_path: Path = spec["path"]

        if not checkpoint_path.is_file():
            logger.error("Checkpoint file not found for %s: %s", model_id, checkpoint_path)
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info("Loading checkpoint for %s from %s", model_id, checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model_config" not in checkpoint_data:
            raise KeyError(f"Checkpoint for {model_id} lacks 'model_config'")

        config = ModelConfig.from_dict(checkpoint_data["model_config"])
        model = MiniLLM(config).to(self.device)
        model.eval()

        checkpoint_info = load_checkpoint(checkpoint_path, model, map_location=self.device)
        total_params = model.count_parameters(trainable_only=False)

        info = ModelInfo(
            model_id=model_id,
            display_name=spec["display_name"],
            description=spec["description"],
            training_subset=spec["training_subset"],
            parameter_count=total_params,
            context_length=config.block_size,
            validation_loss=spec.get("validation_loss", checkpoint_info.get("best_validation_loss")),
            validation_perplexity=spec.get("validation_perplexity"),
            batch_size=spec.get("batch_size"),
            training_steps=spec.get("steps", checkpoint_info.get("step", 10000)),
        )

        self._models[model_id] = LoadedModelEntry(
            model_id=model_id,
            model=model,
            config=config,
            metadata=info,
            checkpoint_step=checkpoint_info.get("step", 10000),
            best_loss=checkpoint_info.get("best_validation_loss"),
        )
        logger.info("Successfully loaded %s (%d parameters) on %s", model_id, total_params, self.device)

    def get_model_entry(self, model_id: str) -> LoadedModelEntry:
        """Retrieve model entry by ID."""
        if not self._is_initialized:
            raise ModelNotLoadedError(model_id, reason="Registry is not yet initialized.")
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        return self._models[model_id]

    def list_models(self) -> List[ModelInfo]:
        """Return list of metadata for all currently registered models."""
        return [entry.metadata for entry in self._models.values()]

    def is_healthy(self) -> bool:
        """Verify at least one model is active."""
        return self._is_initialized and len(self._models) > 0

    @property
    def loaded_model_ids(self) -> List[str]:
        """Return IDs of loaded models."""
        return list(self._models.keys())
