
import torch

from mini_llm.config import ModelConfig
from mini_llm.data import LanguageModelDataset, create_dataloader
from mini_llm.model import MiniLLM
from mini_llm.train import evaluate, train_model
from mini_llm.tokenizer import ByteTokenizer


def test_training_updates_model_and_saves_checkpoint(tmp_path):
    torch.manual_seed(42)

    data_path = tmp_path / "stories.txt"
    data_path.write_text(
        "\n\n<|document_boundary|>\n\n".join(
            [
                "A cat sat.",
                "A dog ran.",
                "The bird flew.",
                "A child played.",
            ]
        ),
        encoding="utf-8",
    )

    config = ModelConfig(
        vocab_size=259,
        block_size=8,
        n_layers=1,
        n_heads=2,
        n_embd=16,
        dropout=0.0,
    )

    dataset = LanguageModelDataset(
        data_path, ByteTokenizer(), config.block_size
    )
    loader = create_dataloader(
        dataset, batch_size=2, shuffle=False
    )

    model = MiniLLM(config)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    result = train_model(
        model,
        loader,
        loader,
        max_steps=2,
        eval_interval=1,
        save_interval=1,
        checkpoint_dir=tmp_path / "checkpoints",
        log_path=tmp_path / "metrics.jsonl",
        device="cpu",
        learning_rate=1e-3,
    )

    assert result["steps"] == 2
    assert (tmp_path / "checkpoints" / "latest.pt").is_file()
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "metrics.jsonl").is_file()

    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )

    metrics = evaluate(model, loader, torch.device("cpu"))
    assert metrics["loss"] > 0
    assert metrics["perplexity"] >= 1
