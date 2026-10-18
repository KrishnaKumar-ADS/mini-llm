
"""Training and evaluation utilities for Mini LLM."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from mini_llm.checkpoint import save_checkpoint
from mini_llm.model import MiniLLM
from mini_llm.tokenizer import PAD_TOKEN_ID


def evaluate(
    model: MiniLLM,
    data_loader: Any,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute token-weighted loss and perplexity.

    If max_batches is supplied, evaluate only that many batches.
    Use full validation for final research measurements.
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches_seen = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            if max_batches is not None and batches_seen >= max_batches:
                break

            batches_seen += 1
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            output = model(inputs, targets)
            loss = output["loss"]

            if loss is None:
                raise RuntimeError("Model did not return a loss")

            valid_tokens = int((targets != PAD_TOKEN_ID).sum().item())

            if valid_tokens:
                total_loss += float(loss.item()) * valid_tokens
                total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError(
            "Evaluation contains no valid target tokens. "
            "Check the dataset and padding configuration."
        )

    mean_loss = total_loss / total_tokens

    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "tokens": float(total_tokens),
        "batches": float(batches_seen),
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a metrics file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def train_model(
    model: MiniLLM,
    train_loader: Any,
    validation_loader: Any,
    *,
    max_steps: int,
    eval_interval: int,
    checkpoint_dir: str | Path,
    log_path: str | Path,
    device: str | torch.device | None = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    save_interval: int = 250,
    eval_max_batches: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train for a fixed number of optimizer steps."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if eval_interval < 1:
        raise ValueError("eval_interval must be at least 1")
    if save_interval < 1:
        raise ValueError("save_interval must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    if eval_max_batches is not None and eval_max_batches < 1:
        raise ValueError("eval_max_batches must be at least 1")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint_dir = Path(checkpoint_dir)
    log_path = Path(log_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    start_time = time.perf_counter()
    running_loss = 0.0
    running_batches = 0
    completed_steps = 0
    train_iterator = iter(train_loader)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Initial validation before any training updates.
    initial_metrics = evaluate(
        model,
        validation_loader,
        device,
        max_batches=eval_max_batches,
    )

    elapsed = time.perf_counter() - start_time
    _append_jsonl(
        log_path,
        {
            "step": 0,
            "split": "validation",
            **initial_metrics,
            "elapsed_seconds": elapsed,
        },
    )

    best_validation_loss = initial_metrics["loss"]

    save_checkpoint(
        checkpoint_dir / "best.pt",
        model,
        optimizer,
        step=0,
        best_validation_loss=best_validation_loss,
        metadata=metadata,
    )

    print(
        f"Initial validation | loss {initial_metrics['loss']:.4f} | "
        f"perplexity {initial_metrics['perplexity']:.2f} | "
        f"batches {int(initial_metrics['batches'])}"
    )

    while completed_steps < max_steps:
        model.train()

        try:
            inputs, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)

            try:
                inputs, targets = next(train_iterator)
            except StopIteration as exc:
                raise ValueError("Training DataLoader is empty") from exc

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(inputs, targets)
        loss = output["loss"]

        if loss is None:
            raise RuntimeError("Model did not return a training loss")

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at step {completed_steps + 1}"
            )

        loss.backward()

        gradient_norm = clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )

        optimizer.step()

        completed_steps += 1
        running_loss += float(loss.detach().item())
        running_batches += 1

        should_evaluate = (
            completed_steps % eval_interval == 0
            or completed_steps == max_steps
        )

        if should_evaluate:
            train_loss = running_loss / running_batches

            validation_metrics = evaluate(
                model,
                validation_loader,
                device,
                max_batches=eval_max_batches,
            )

            elapsed = time.perf_counter() - start_time

            train_record = {
                "step": completed_steps,
                "split": "training",
                "loss": train_loss,
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": elapsed,
            }

            if device.type == "cuda":
                train_record["peak_gpu_memory_bytes"] = (
                    torch.cuda.max_memory_allocated(device)
                )

            _append_jsonl(log_path, train_record)

            _append_jsonl(
                log_path,
                {
                    "step": completed_steps,
                    "split": "validation",
                    **validation_metrics,
                    "elapsed_seconds": elapsed,
                },
            )

            memory_text = ""
            if device.type == "cuda":
                peak_gb = (
                    torch.cuda.max_memory_allocated(device)
                    / (1024 ** 3)
                )
                memory_text = f" | peak GPU {peak_gb:.2f} GB"

            print(
                f"Step {completed_steps:>6}/{max_steps} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {validation_metrics['loss']:.4f} | "
                f"val ppl {validation_metrics['perplexity']:.2f} | "
                f"time {elapsed:.1f}s"
                f"{memory_text}"
            )

            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]

                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    step=completed_steps,
                    best_validation_loss=best_validation_loss,
                    metadata=metadata,
                )

            running_loss = 0.0
            running_batches = 0

        if completed_steps % save_interval == 0:
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                step=completed_steps,
                best_validation_loss=best_validation_loss,
                metadata=metadata,
            )

    save_checkpoint(
        checkpoint_dir / "latest.pt",
        model,
        optimizer,
        step=completed_steps,
        best_validation_loss=best_validation_loss,
        metadata=metadata,
    )

    return {
        "steps": completed_steps,
        "best_validation_loss": best_validation_loss,
        "elapsed_seconds": time.perf_counter() - start_time,
        "device": str(device),
    }
