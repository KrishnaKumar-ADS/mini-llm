
import pytest
import torch

from mini_llm.config import ModelConfig
from mini_llm.model import MiniLLM, RMSNorm
from mini_llm.tokenizer import PAD_TOKEN_ID


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


def test_model_output_shape(config):
    model = MiniLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids)

    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["loss"] is None


def test_model_calculates_next_token_loss(config):
    model = MiniLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids, targets)

    assert output["loss"] is not None
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])


def test_padding_targets_are_ignored(config):
    model = MiniLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    targets[:, -2:] = PAD_TOKEN_ID

    output = model(input_ids, targets)

    assert torch.isfinite(output["loss"])


def test_model_supports_backpropagation(config):
    model = MiniLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids, targets)
    output["loss"].backward()

    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_model_is_causally_masked(config):
    torch.manual_seed(23)
    model = MiniLLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    modified_ids = input_ids.clone()
    modified_ids[:, 5:] = torch.randint(
        0, config.vocab_size, modified_ids[:, 5:].shape
    )

    with torch.no_grad():
        original_logits = model(input_ids)["logits"]
        modified_logits = model(modified_ids)["logits"]

    torch.testing.assert_close(
        original_logits[:, :5, :],
        modified_logits[:, :5, :],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
def test_model_supports_normalization_variants(config, norm_type):
    config.norm_type = norm_type
    model = MiniLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(input_ids)["logits"]

    assert logits.shape == (2, 8, config.vocab_size)


@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_model_supports_activation_variants(config, activation):
    config.activation = activation
    model = MiniLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(input_ids)["logits"]

    assert logits.shape == (2, 8, config.vocab_size)


@pytest.mark.parametrize("positional_encoding", ["learned", "sinusoidal"])
def test_model_supports_position_variants(config, positional_encoding):
    config.positional_encoding = positional_encoding
    model = MiniLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(input_ids)["logits"]

    assert logits.shape == (2, 8, config.vocab_size)


def test_rope_is_not_silently_enabled(config):
    config.positional_encoding = "rope"

    with pytest.raises(NotImplementedError, match="RoPE"):
        MiniLLM(config)


def test_invalid_token_ids_are_rejected(config):
    model = MiniLLM(config)
    input_ids = torch.tensor([[0, 1, config.vocab_size]])

    with pytest.raises(ValueError, match="outside vocabulary"):
        model(input_ids)


def test_invalid_input_dtype_is_rejected(config):
    model = MiniLLM(config)
    input_ids = torch.randn(1, 8)

    with pytest.raises(TypeError, match="torch.long"):
        model(input_ids)


def test_parameter_count_is_positive(config):
    model = MiniLLM(config)

    assert model.count_parameters() > 0
    assert model.count_parameters(trainable_only=False) >= (
        model.count_parameters()
    )


def test_token_embeddings_are_tied_to_output_head(config):
    model = MiniLLM(config)

    assert (
        model.token_embedding.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    )


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(32)
    x = torch.randn(2, 8, 32)

    assert norm(x).shape == x.shape
