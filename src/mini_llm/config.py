
"""Validated configuration for the Mini LLM Laboratory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Settings that define a decoder-only transformer."""

    vocab_size: int = 259
    block_size: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True
    mlp_ratio: int = 4

    # Architecture settings for controlled experiments.
    activation: str = "gelu"
    norm_type: str = "layernorm"
    positional_encoding: str = "learned"
    n_kv_heads: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.n_embd // self.n_heads

    @property
    def effective_n_kv_heads(self) -> int:
        """Number of key/value heads; defaults to standard MHA."""
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def attention_type(self) -> str:
        """Describe the attention layout implied by the KV head count."""
        kv_heads = self.effective_n_kv_heads

        if kv_heads == self.n_heads:
            return "mha"
        if kv_heads == 1:
            return "mqa"
        return "gqa"

    @property
    def mlp_hidden_dim(self) -> int:
        """Hidden dimension of a standard feed-forward network."""
        return self.n_embd * self.mlp_ratio

    def validate(self) -> None:
        """Raise clear errors for unsupported or inconsistent settings."""
        positive_int_fields = (
            "vocab_size",
            "block_size",
            "n_layers",
            "n_heads",
            "n_embd",
            "mlp_ratio",
        )

        for name in positive_int_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        if self.n_embd % self.n_heads != 0:
            raise ValueError(
                "n_embd must be divisible by n_heads; "
                f"received n_embd={self.n_embd}, n_heads={self.n_heads}"
            )

        if self.n_kv_heads is not None:
            if (
                not isinstance(self.n_kv_heads, int)
                or isinstance(self.n_kv_heads, bool)
            ):
                raise TypeError("n_kv_heads must be an integer or None")

            if not 1 <= self.n_kv_heads <= self.n_heads:
                raise ValueError(
                    "n_kv_heads must be between 1 and n_heads"
                )

            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError(
                    "n_heads must be divisible by n_kv_heads"
                )

        if not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be numeric")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        if not isinstance(self.bias, bool):
            raise TypeError("bias must be a boolean")

        allowed_activations = {"gelu", "swiglu"}
        if self.activation not in allowed_activations:
            raise ValueError(
                f"activation must be one of {sorted(allowed_activations)}"
            )

        allowed_norms = {"layernorm", "rmsnorm"}
        if self.norm_type not in allowed_norms:
            raise ValueError(
                f"norm_type must be one of {sorted(allowed_norms)}"
            )

        allowed_positions = {"learned", "sinusoidal", "rope"}
        if self.positional_encoding not in allowed_positions:
            raise ValueError(
                "positional_encoding must be one of "
                f"{sorted(allowed_positions)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Save this configuration as formatted JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        """Build a config from a dictionary, rejecting unknown keys."""
        if not isinstance(values, dict):
            raise TypeError("Configuration must be a JSON object")

        allowed_keys = {field.name for field in fields(cls)}
        unknown_keys = set(values) - allowed_keys

        if unknown_keys:
            raise ValueError(
                f"Unknown configuration keys: {sorted(unknown_keys)}"
            )

        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """Load and validate a configuration from a JSON file."""
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file: {path}") from exc

        return cls.from_dict(values)
