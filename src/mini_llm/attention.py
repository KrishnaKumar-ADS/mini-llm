
"""Causal self-attention for decoder-only Transformers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mini_llm.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head, grouped-query, or multi-query causal self-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.effective_n_kv_heads
        self.head_dim = config.head_dim
        self.block_size = config.block_size
        self.dropout = config.dropout

        self.q_proj = nn.Linear(
            config.n_embd, self.n_heads * self.head_dim, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias
        )
        self.out_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.

        Args:
            x: Tensor shaped (batch_size, sequence_length, embedding_dim).

        Returns:
            Tensor shaped (batch_size, sequence_length, embedding_dim).
        """
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, sequence, embedding); "
                f"received {tuple(x.shape)}"
            )

        batch_size, sequence_length, _ = x.shape

        if sequence_length == 0:
            raise ValueError("sequence_length must be greater than zero")

        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"block_size {self.block_size}"
            )

        # (B, T, C) -> (B, heads, T, head_dim)
        q = self.q_proj(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)

        # Keys and values may have fewer heads than queries for GQA/MQA.
        k = self.k_proj(x).view(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Expand KV heads to match query heads when using GQA or MQA.
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # is_causal=True blocks access to future positions.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # (B, heads, T, head_dim) -> (B, T, C)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.n_heads * self.head_dim
        )

        return self.residual_dropout(self.out_proj(attended))
