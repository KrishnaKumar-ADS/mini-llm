
"""Save and restore Mini LLM training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mini_llm.config import ModelConfig
from mini_llm.model import MiniLLM


def save_checkpoint(
    path: str | Path,
    model: MiniLLM,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int = 0,
    best_validation_loss: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model weights, optimizer state, and reproducibility metadata."""
    if step < 0:
        raise ValueError("step cannot be negative")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "step": step,
        "best_validation_loss": best_validation_loss,
        "metadata": metadata or {},
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: MiniLLM,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Restore a checkpoint into an existing model and optional optimizer."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format version")

    saved_config = ModelConfig.from_dict(checkpoint["model_config"])

    if saved_config.to_dict() != model.config.to_dict():
        raise ValueError(
            "Checkpoint model configuration does not match the supplied model"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None:
        if optimizer_state is None:
            raise ValueError(
                "Checkpoint has no optimizer state to restore"
            )
        optimizer.load_state_dict(optimizer_state)

    if restore_rng:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

        cuda_rng_states = checkpoint.get("cuda_rng_states")
        if cuda_rng_states is not None and torch.cuda.is_available():
            if len(cuda_rng_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_rng_states)

    return {
        "step": checkpoint["step"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "metadata": checkpoint.get("metadata", {}),
    }
