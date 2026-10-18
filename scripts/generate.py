
"""Generate text from a saved Mini LLM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.generation import generate_text
from mini_llm.model import MiniLLM
from mini_llm.tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Mini LLM checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "experiments"
        / "overfit_test"
        / "checkpoints"
        / "best.pt",
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The cat",
        help="Text prompt to continue.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run the overfit test first or provide --checkpoint."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    # Read the configuration saved with the checkpoint.
        # Read the configuration saved with the checkpoint.
        # Read the configuration saved with the checkpoint.
    checkpoint_data = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "model_config" not in checkpoint_data:
        raise KeyError(
            "The checkpoint does not contain 'model_config'. "
            f"Available keys: {list(checkpoint_data.keys())}"
        )

    config = ModelConfig.from_dict(checkpoint_data["model_config"])
    model = MiniLLM(config).to(device)

    # Restore the learned model weights.
    checkpoint_info = load_checkpoint(
        args.checkpoint,
        model,
        map_location=device,
    )

    tokenizer = ByteTokenizer()

    print("=" * 64)
    print("MINI LLM TEXT GENERATION")
    print("=" * 64)
    print(f"Checkpoint:    {args.checkpoint.resolve()}")
    print(f"Device:        {device}")
    if device.type == "cuda":
        print(f"GPU:           {torch.cuda.get_device_name(device)}")
    print(f"Training step: {checkpoint_info.get('step', 'unknown')}")
    print(f"Parameters:    {model.count_parameters():,}")
    print(f"Temperature:   {args.temperature}")
    print(f"Top-k:         {args.top_k}")
    print(f"Top-p:         {args.top_p}")
    print("-" * 64)
    print(f"Prompt: {args.prompt}")
    print("-" * 64)

    generated = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        seed=args.seed,
    )

    print("GENERATED TEXT")
    print("-" * 64)
    print(generated)
    print("=" * 64)


if __name__ == "__main__":
    main()
