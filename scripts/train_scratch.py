
"""Train the scratch-built Mini LLM."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mini_llm.config import ModelConfig
from mini_llm.data import LanguageModelDataset, create_dataloader
from mini_llm.model import MiniLLM
from mini_llm.tokenizer import ByteTokenizer
from mini_llm.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Mini LLM from scratch."
    )

    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument(
        "--train-file",
        default="data/subsets/train_25k.txt",
    )
    parser.add_argument(
        "--validation-file",
        default="data/validation.txt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help=(
            "Maximum validation batches per evaluation. "
            "Omit for full validation."
        ),
    )
    parser.add_argument(
        "--run-name",
        default="tiny_baseline",
        help="Name for checkpoint and metrics directories.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.eval_interval < 1:
        raise ValueError("--eval-interval must be at least 1")
    if args.save_interval < 1:
        raise ValueError("--save-interval must be at least 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.eval_batches is not None and args.eval_batches < 1:
        raise ValueError("--eval-batches must be at least 1")
    if not args.run_name.strip():
        raise ValueError("--run-name cannot be empty")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config_path = PROJECT_ROOT / args.config
    train_path = PROJECT_ROOT / args.train_file
    validation_path = PROJECT_ROOT / args.validation_file

    config = ModelConfig.from_json(config_path)
    tokenizer = ByteTokenizer()

    train_dataset = LanguageModelDataset(
        train_path,
        tokenizer,
        config.block_size,
    )
    validation_dataset = LanguageModelDataset(
        validation_path,
        tokenizer,
        config.block_size,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = MiniLLM(config)

    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "Not used"
    )

    print("=" * 64)
    print("MINI LLM TRAINING")
    print("=" * 64)
    print(f"Device:             {device}")
    print(f"GPU:                {gpu_name}")
    print(f"Config:             {config_path}")
    print(f"Training file:      {train_path}")
    print(f"Validation file:    {validation_path}")
    print(f"Training windows:   {len(train_dataset):,}")
    print(f"Validation windows: {len(validation_dataset):,}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Maximum steps:      {args.max_steps:,}")
    print(f"Evaluation batches: {args.eval_batches or 'Full validation'}")
    print(f"Parameters:         {model.count_parameters():,}")
    print("=" * 64)

    run_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "experiments"
        / args.run_name
    )

    metrics_path = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / args.run_name
        / "training_metrics.jsonl"
    )

    result = train_model(
        model,
        train_loader,
        validation_loader,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=run_dir / "checkpoints",
        log_path=metrics_path,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_max_batches=args.eval_batches,
        metadata={
            "run_name": args.run_name,
            "seed": args.seed,
            "config_path": str(config_path.relative_to(PROJECT_ROOT)),
            "train_file": str(train_path.relative_to(PROJECT_ROOT)),
            "validation_file": str(
                validation_path.relative_to(PROJECT_ROOT)
            ),
            "training_windows": len(train_dataset),
            "validation_windows": len(validation_dataset),
            "batch_size": args.batch_size,
            "eval_max_batches": args.eval_batches,
        },
    )

    print("\nTraining finished.")
    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"Checkpoints: {run_dir / 'checkpoints'}")
    print(f"Metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
