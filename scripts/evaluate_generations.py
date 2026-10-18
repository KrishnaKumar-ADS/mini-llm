
"""Compare story generation from the existing 25k and 100k Mini LLM models."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_SCRIPT = PROJECT_ROOT / "scripts" / "generate.py"

MODELS = {
    "25k": (
        PROJECT_ROOT
        / "artifacts/experiments/tiny_25k_baseline_b288_v1"
        / "checkpoints/best.pt"
    ),
    "100k": (
        PROJECT_ROOT
        / "artifacts/experiments/tiny_100k_b1280_v1"
        / "checkpoints/best.pt"
    ),
}

# The same prompts are used for both models.
PROMPTS = [
    "Once upon a time",
    "A little girl found",
    "One sunny morning",
    "A young boy went into the forest",
    "The little dog was hungry",
    "One day, a rabbit met a turtle",
    "It was raining outside, so",
    "The old man opened the door and",
    "Lucy wanted to help her friend",
    "After school, the children",
]

# Fixed settings make the comparison reproducible.
SEED = 42
MAX_NEW_TOKENS = 1000
TEMPERATURE = 0.8
TOP_K = 20
TOP_P = 0.9
DEVICE = "cuda"


def extract_generated_text(stdout: str) -> str:
    """Extract generated text from generate.py's terminal output."""
    marker = "GENERATED TEXT"

    if marker not in stdout:
        raise ValueError(
            "Could not find the GENERATED TEXT marker in model output."
        )

    lines = stdout.split(marker, maxsplit=1)[1].splitlines()

    # Skip the separator immediately below the marker.
    while lines and (
        not lines[0].strip()
        or set(lines[0].strip()) == {"-"}
    ):
        lines.pop(0)

    generated_lines = []

    for line in lines:
        stripped = line.strip()

        # Stop at the terminal's closing header separator.
        if stripped and set(stripped) == {"="}:
            break

        generated_lines.append(line)

    return "\n".join(generated_lines).strip()


def main() -> None:
    if not GENERATION_SCRIPT.exists():
        raise FileNotFoundError(
            f"Generation script not found: {GENERATION_SCRIPT}"
        )

    for name, checkpoint in MODELS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {checkpoint}"
            )

    output_dir = PROJECT_ROOT / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "generation_comparison.csv"
    total_runs = len(MODELS) * len(PROMPTS)
    completed = 0

    print("=" * 72)
    print("MINI LLM GENERATION EVALUATION")
    print("=" * 72)
    print(f"Models: {len(MODELS)}")
    print(f"Prompts per model: {len(PROMPTS)}")
    print(f"Total generations: {total_runs}")
    print(f"Device: {DEVICE}")
    print(f"Output: {output_csv}")
    print("=" * 72)

    with output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        fieldnames = [
            "model",
            "checkpoint",
            "prompt",
            "seed",
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "device",
            "generation_seconds",
            "generated_text",
        ]

        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames
        )
        writer.writeheader()

        for model_name, checkpoint in MODELS.items():
            for prompt_index, prompt in enumerate(PROMPTS, start=1):
                command = [
                    sys.executable,
                    str(GENERATION_SCRIPT),
                    "--checkpoint", str(checkpoint),
                    "--prompt", prompt,
                    "--max-new-tokens", str(MAX_NEW_TOKENS),
                    "--temperature", str(TEMPERATURE),
                    "--top-k", str(TOP_K),
                    "--top-p", str(TOP_P),
                    "--seed", str(SEED),
                    "--device", DEVICE,
                ]

                print(
                    f"[{completed + 1}/{total_runs}] "
                    f"Model={model_name}, "
                    f"Prompt={prompt_index}/{len(PROMPTS)}: "
                    f"{prompt!r}"
                )

                start = time.perf_counter()

                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )

                elapsed = time.perf_counter() - start

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Generation failed for {model_name}, "
                        f"prompt {prompt_index}.\n"
                        f"STDOUT:\n{result.stdout}\n"
                        f"STDERR:\n{result.stderr}"
                    )

                generated_text = extract_generated_text(
                    result.stdout
                )

                if not generated_text:
                    raise RuntimeError(
                        f"Empty generation for {model_name}, "
                        f"prompt {prompt_index}."
                    )

                writer.writerow(
                    {
                        "model": model_name,
                        "checkpoint": str(checkpoint.relative_to(
                            PROJECT_ROOT
                        )),
                        "prompt": prompt,
                        "seed": SEED,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": TEMPERATURE,
                        "top_k": TOP_K,
                        "top_p": TOP_P,
                        "device": DEVICE,
                        "generation_seconds": round(elapsed, 3),
                        "generated_text": generated_text,
                    }
                )

                # Ensure completed results are saved incrementally.
                csv_file.flush()

                completed += 1
                print(f"  Completed in {elapsed:.2f}s")
                print(f"  Output: {generated_text[:160]}")
                print()

    print("=" * 72)
    print("EVALUATION COMPLETED")
    print(f"Generations saved: {completed}")
    print(f"Results: {output_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
