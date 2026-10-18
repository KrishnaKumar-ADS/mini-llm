
import pytest
import torch

from mini_llm.checkpoint import load_checkpoint, save_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model import MiniLLM


@pytest.fixture
def config():
    return ModelConfig(
        block_size=8,
        n_layers=1,
        n_heads=2,
        n_embd=16,
        dropout=0.0,
    )


def test_checkpoint_preserves_model_logits(tmp_path, config):
    torch.manual_seed(17)
    model = MiniLLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    with torch.no_grad():
        expected = model(input_ids)["logits"]

    path = tmp_path / "checkpoints" / "model.pt"
    save_checkpoint(path, model, step=12)

    restored_model = MiniLLM(config).eval()
    info = load_checkpoint(path, restored_model)

    with torch.no_grad():
        actual = restored_model(input_ids)["logits"]

    torch.testing.assert_close(actual, expected)
    assert info["step"] == 12


def test_checkpoint_restores_optimizer(tmp_path, config):
    model = MiniLLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    targets = torch.randint(0, config.vocab_size, (1, 6))

    model(input_ids, targets)["loss"].backward()
    optimizer.step()

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, step=1)

    restored_model = MiniLLM(config)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=1e-3
    )

    info = load_checkpoint(
        path, restored_model, restored_optimizer
    )

    assert info["step"] == 1
    assert len(restored_optimizer.state) > 0


def test_missing_checkpoint_fails(tmp_path, config):
    model = MiniLLM(config)

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model)


def test_mismatched_config_fails(tmp_path, config):
    model = MiniLLM(config)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model)

    incompatible = ModelConfig(
        block_size=12,
        n_layers=1,
        n_heads=2,
        n_embd=16,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, MiniLLM(incompatible))
