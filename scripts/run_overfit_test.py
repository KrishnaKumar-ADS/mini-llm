
"""Run a controlled tiny-dataset overfit experiment."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mini_llm.config import ModelConfig
from mini_llm.data import (
    DOCUMENT_SEPARATOR,
    LanguageModelDataset,
    create_dataloader,
)
from mini_llm.model import MiniLLM
from mini_llm.tokenizer import ByteTokenizer
from mini_llm.train import train_model


def main() -> None:
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Small, fixed dataset deliberately reused for both training
    # and validation. This is an overfit diagnostic, not evaluation.
    stories = [
        "The cat sat on a red mat.",
        "The dog ran to the big tree.",
        "A bird sang in the morning.",
        "The little fox found a hat.",
    ]

    run_dir = PROJECT_ROOT / "artifacts" / "experiments" / "overfit_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = run_dir / "overfit_stories.txt"
    data_path.write_text(
        DOCUMENT_SEPARATOR.join(stories),
        encoding="utf-8",
    )

    config = ModelConfig.from_json(PROJECT_ROOT / "configs" / "tiny.json")
    tokenizer = ByteTokenizer()

    dataset = LanguageModelDataset(
        data_path,
        tokenizer,
        block_size=config.block_size,
    )

    train_loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        seed=seed,
    )

    # Reusing the same data is intentional for this diagnostic only.
    validation_loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        seed=seed,
    )

    model = MiniLLM(config)

    metrics_path = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / "overfit_test"
        / "training_metrics.jsonl"
    )

    print("=" * 64)
    print("MINI LLM OVERFIT TEST")
    print("=" * 64)
    print(f"Device:           {device}")
    if device.type == "cuda":
        print(f"GPU:              {torch.cuda.get_device_name(device)}")
    print(f"Stories:          {len(stories)}")
    print(f"Training windows: {len(dataset)}")
    print(f"Batch size:       4")
    print(f"Model parameters: {model.count_parameters():,}")
    print("Training and validation use the same four stories.")
    print("=" * 64)

    result = train_model(
        model,
        train_loader,
        validation_loader,
        max_steps=200,
        eval_interval=20,
        save_interval=50,
        checkpoint_dir=run_dir / "checkpoints",
        log_path=metrics_path,
        device=device,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        metadata={
            "experiment": "overfit_test",
            "seed": seed,
            "story_count": len(stories),
            "purpose": "Verify the model can memorize a tiny dataset",
        },
    )

    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    validation_records = [
        record for record in records
        if record.get("split") == "validation"
    ]

    initial_loss = validation_records[0]["loss"]
    final_loss = validation_records[-1]["loss"]

    print("\nOVERFIT TEST SUMMARY")
    print("-" * 64)
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Loss change:  {initial_loss - final_loss:.4f}")
    print(f"Steps:        {result['steps']}")
    print(f"Device:       {result['device']}")
    print(f"Checkpoints:  {run_dir / 'checkpoints'}")
    print(f"Metrics:      {metrics_path}")

    if final_loss < initial_loss:
        print("\nPASS: Loss decreased on the tiny dataset.")
        print("Inspect the loss curve to assess the degree of memorization.")
    else:
        print("\nWARNING: Loss did not decrease.")
        print("Inspect the training logs before proceeding.")


if __name__ == "__main__":
    main()
