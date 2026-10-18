
import json

import pytest

from mini_llm.config import ModelConfig


def test_default_config_is_valid():
    config = ModelConfig()

    assert config.vocab_size == 259
    assert config.block_size == 128
    assert config.n_layers == 4
    assert config.n_heads == 4
    assert config.n_embd == 256
    assert config.head_dim == 64
    assert config.mlp_hidden_dim == 1024
    assert config.attention_type == "mha"


def test_tiny_config_loads():
    config = ModelConfig.from_json("configs/tiny.json")

    assert config.n_layers == 4
    assert config.n_embd == 256
    assert config.attention_type == "mha"


def test_small_config_loads():
    config = ModelConfig.from_json("configs/small.json")

    assert config.n_layers == 6
    assert config.n_heads == 6
    assert config.n_embd == 384
    assert config.head_dim == 64


def test_invalid_embedding_dimension_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=250, n_heads=4)


@pytest.mark.parametrize("block_size", [0, -1])
def test_invalid_block_size_is_rejected(block_size):
    with pytest.raises(ValueError, match="block_size"):
        ModelConfig(block_size=block_size)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_invalid_dropout_is_rejected(dropout):
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=dropout)


def test_invalid_attention_head_configuration_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_heads=8, n_embd=256, n_kv_heads=3)


def test_gqa_and_mqa_are_identified():
    gqa = ModelConfig(n_heads=8, n_embd=256, n_kv_heads=2)
    mqa = ModelConfig(n_heads=8, n_embd=256, n_kv_heads=1)

    assert gqa.attention_type == "gqa"
    assert mqa.attention_type == "mqa"


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        ModelConfig.from_dict({"unknown_setting": 123})


def test_config_save_and_load_round_trip(tmp_path):
    original = ModelConfig(n_layers=3, n_embd=192, n_heads=3)
    path = tmp_path / "model_config.json"

    original.save_json(path)
    restored = ModelConfig.from_json(path)

    assert restored.to_dict() == original.to_dict()


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        ModelConfig.from_json(path)
