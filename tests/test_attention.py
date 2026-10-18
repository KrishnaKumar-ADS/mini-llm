
import pytest
import torch
from torch.nn import functional as F

from mini_llm.attention import CausalSelfAttention
from mini_llm.config import ModelConfig


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=259,
        block_size=16,
        n_layers=2,
        n_heads=4,
        n_embd=32,
        dropout=0.0,
    )


def test_attention_output_shape(config):
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 8, config.n_embd)

    output = attention(x)

    assert output.shape == x.shape


def test_attention_supports_batch_size_one(config):
    attention = CausalSelfAttention(config)
    x = torch.randn(1, 5, config.n_embd)

    assert attention(x).shape == (1, 5, config.n_embd)


def test_attention_cannot_see_future_tokens(config):
    torch.manual_seed(7)
    attention = CausalSelfAttention(config)
    attention.eval()

    original = torch.randn(1, 8, config.n_embd)
    modified = original.clone()

    # Change tokens at positions 5, 6 and 7.
    modified[:, 5:, :] = torch.randn_like(modified[:, 5:, :]) * 10

    with torch.no_grad():
        original_output = attention(original)
        modified_output = attention(modified)

    # Outputs at positions 0 through 4 must remain unchanged.
    torch.testing.assert_close(
        original_output[:, :5, :],
        modified_output[:, :5, :],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_mha_gqa_and_mqa_output_shapes(n_kv_heads):
    config = ModelConfig(
        block_size=16,
        n_layers=2,
        n_heads=4,
        n_embd=32,
        n_kv_heads=n_kv_heads,
        dropout=0.0,
    )
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 8, 32)

    output = attention(x)

    assert output.shape == x.shape


def test_invalid_input_dimensions_are_rejected(config):
    attention = CausalSelfAttention(config)

    with pytest.raises(ValueError, match="Expected input shape"):
        attention(torch.randn(2, 32))


def test_sequence_longer_than_block_size_is_rejected(config):
    attention = CausalSelfAttention(config)
    x = torch.randn(1, 17, config.n_embd)

    with pytest.raises(ValueError, match="exceeds block_size"):
        attention(x)


def test_empty_sequence_is_rejected(config):
    attention = CausalSelfAttention(config)
    x = torch.randn(1, 0, config.n_embd)

    with pytest.raises(ValueError, match="sequence_length"):
        attention(x)


def test_fused_attention_matches_manual_causal_attention(config):
    torch.manual_seed(11)
    attention = CausalSelfAttention(config)
    attention.eval()

    x = torch.randn(2, 6, config.n_embd)

    with torch.no_grad():
        actual = attention(x)

        batch_size, sequence_length, _ = x.shape

        q = attention.q_proj(x).view(
            batch_size, sequence_length,
            config.n_heads, config.head_dim
        ).transpose(1, 2)

        k = attention.k_proj(x).view(
            batch_size, sequence_length,
            config.n_heads, config.head_dim
        ).transpose(1, 2)

        v = attention.v_proj(x).view(
            batch_size, sequence_length,
            config.n_heads, config.head_dim
        ).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (config.head_dim ** 0.5)

        # True means that attention to this position is forbidden.
        future_mask = torch.triu(
            torch.ones(
                sequence_length, sequence_length, dtype=torch.bool
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(future_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        manual = weights @ v

        manual = manual.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, config.n_embd
        )
        manual = attention.out_proj(manual)

    torch.testing.assert_close(actual, manual, rtol=1e-5, atol=1e-6)
