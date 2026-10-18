
"""Decoder-only Transformer language model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from mini_llm.attention import CausalSelfAttention
from mini_llm.config import ModelConfig
from mini_llm.tokenizer import PAD_TOKEN_ID


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(rms + self.eps)
        return x_normalized * self.weight


def make_norm(norm_type: str, dim: int) -> nn.Module:
    """Construct the normalization layer selected by the config."""
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unsupported normalization type: {norm_type}")


class FeedForward(nn.Module):
    """Transformer feed-forward network with GELU or SwiGLU."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = config.mlp_hidden_dim

        if config.activation == "gelu":
            self.network = nn.Sequential(
                nn.Linear(config.n_embd, hidden_dim, bias=config.bias),
                nn.GELU(),
                nn.Linear(hidden_dim, config.n_embd, bias=config.bias),
            )
        elif config.activation == "swiglu":
            self.gate_proj = nn.Linear(
                config.n_embd, hidden_dim, bias=config.bias
            )
            self.up_proj = nn.Linear(
                config.n_embd, hidden_dim, bias=config.bias
            )
            self.down_proj = nn.Linear(
                hidden_dim, config.n_embd, bias=config.bias
            )
        else:
            raise ValueError(
                f"Unsupported activation: {config.activation}"
            )

        self.activation = config.activation
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            output = self.network(x)
        else:
            output = self.down_proj(
                F.silu(self.gate_proj(x)) * self.up_proj(x)
            )

        return self.dropout(output)


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer block with residual connections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.norm1 = make_norm(config.norm_type, config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm2 = make_norm(config.norm_type, config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class MiniLLM(nn.Module):
    """Small decoder-only Transformer for next-token prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()

        if config.positional_encoding == "rope":
            raise NotImplementedError(
                "RoPE is configured but not yet implemented in attention.py. "
                "Use learned or sinusoidal positions until that update."
            )

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd
        )

        if config.positional_encoding == "learned":
            self.position_embedding = nn.Embedding(
                config.block_size, config.n_embd
            )
            self.register_buffer(
                "sinusoidal_positions", None, persistent=False
            )
        elif config.positional_encoding == "sinusoidal":
            self.position_embedding = None
            self.register_buffer(
                "sinusoidal_positions",
                self._make_sinusoidal_positions(
                    config.block_size, config.n_embd
                ),
                persistent=False,
            )
        else:
            raise ValueError(
                "Unsupported positional encoding: "
                f"{config.positional_encoding}"
            )

        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = make_norm(config.norm_type, config.n_embd)

        # Project each hidden representation to vocabulary logits.
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )

        # Tie input and output token embeddings to reduce parameters.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._initialize_weights)

    @staticmethod
    def _make_sinusoidal_positions(
        block_size: int, n_embd: int
    ) -> torch.Tensor:
        """Create fixed sinusoidal position vectors."""
        positions = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)

        even_dims = torch.arange(0, n_embd, 2, dtype=torch.float32)
        frequencies = torch.exp(
            -math.log(10000.0) * even_dims / n_embd
        )

        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.zeros(block_size, n_embd)

        encoding[:, 0::2] = torch.sin(angles)
        odd_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(angles[:, :odd_width])

        return encoding.unsqueeze(0)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return vocabulary logits and optional next-token loss.

        Args:
            input_ids: Long tensor shaped (batch, sequence).
            targets: Optional long tensor with the same shape. Each target
                is the desired next token for the corresponding input.

        Returns:
            A dictionary containing logits shaped (batch, sequence, vocab)
            and loss, which is None when targets are not supplied.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch, sequence)"
            )

        batch_size, sequence_length = input_ids.shape

        if batch_size < 1 or sequence_length < 1:
            raise ValueError("Batch and sequence dimensions must be non-empty")

        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"block_size {self.config.block_size}"
            )

        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long")

        if torch.any(input_ids < 0) or torch.any(
            input_ids >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain token IDs outside vocabulary")

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if targets.dtype != torch.long:
                raise TypeError("targets must have dtype torch.long")

            valid_targets = targets != PAD_TOKEN_ID
            if torch.any(valid_targets & (targets < 0)) or torch.any(
                valid_targets & (targets >= self.config.vocab_size)
            ):
                raise ValueError("targets contain invalid token IDs")

        positions = torch.arange(
            sequence_length, device=input_ids.device
        )

        x = self.token_embedding(input_ids)

        if self.position_embedding is not None:
            x = x + self.position_embedding(positions).unsqueeze(0)
        else:
            x = x + self.sinusoidal_positions[:, :sequence_length, :].to(
                device=x.device, dtype=x.dtype
            )

        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )

        return {"logits": logits, "loss": loss}

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
