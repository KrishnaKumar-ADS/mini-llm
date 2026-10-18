
"""Analyze Mini LLM generation samples and validation logs."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "reports" / "evaluation"

GENERATIONS_FILE = EVALUATION_DIR / "generation_comparison.csv"

EXPERIMENTS = {
    "25k": {
        "run": "tiny_25k_baseline_b288_v1",
        "batch_size": 288,
    },
    "100k": {
        "run": "tiny_100k_b1280_v1",
        "batch_size": 1280,
    },
}


def tokenize(text: str) -> list[str]:
    """Extract simple lowercase word-like units."""
    return re.findall(r"\b[\w']+\b", text.lower())


def text_metrics(text: str) -> dict:
    """Calculate diagnostic text statistics, not a quality score."""
    words = tokenize(text)
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]

    trigrams = [
        tuple(words[i:i + 3])
        for i in range(max(0, len(words) - 2))
    ]

    trigram_counts = Counter(trigrams)
    repeated_trigram_occurrences = sum(
        count - 1 for count in trigram_counts.values()
    )

    sentence_counts = Counter(
        re.sub(r"\s+", " ", s.lower()).strip()
        for s in sentences
    )
    repeated_sentences = sum(
        count - 1 for count in sentence_counts.values()
    )

    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "unique_word_ratio": (
            len(set(words)) / len(words) if words else 0.0
        ),
        "repeated_trigram_ratio": (
            repeated_trigram_occurrences / len(trigrams)
            if trigrams else 0.0
        ),
        "repeated_sentence_count": repeated_sentences,
    }


def read_validation_log(run_name: str) -> dict:
    path = (
        ROOT / "reports" / "experiments" / run_name
        / "training_metrics.jsonl"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing training log: {path}")

    validation_rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if record.get("split") == "validation":
                validation_rows.append(record)

    if not validation_rows:
        raise ValueError(f"No validation records found in {path}")

    best = min(validation_rows, key=lambda r: r["loss"])
    latest = max(validation_rows, key=lambda r: r["step"])

    return {
        "run_name": run_name,
        "best_step": best["step"],
        "best_validation_loss": best["loss"],
        "best_validation_perplexity": best.get("perplexity"),
        "latest_step": latest["step"],
        "latest_validation_loss": latest["loss"],
        "latest_validation_perplexity": latest.get("perplexity"),
        "elapsed_seconds_at_latest_eval": latest.get(
            "elapsed_seconds"
        ),
        "validation_evaluations": len(validation_rows),
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> None:
    if not GENERATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Run the generation evaluation first: {GENERATIONS_FILE}"
        )

    with GENERATIONS_FILE.open(
        newline="", encoding="utf-8"
    ) as f:
        generations = list(csv.DictReader(f))

    if not generations:
        raise ValueError("The generation CSV contains no rows.")

    per_sample_path = EVALUATION_DIR / "per_sample_text_metrics.csv"
    summary_path = EVALUATION_DIR / "model_comparison.json"
    validation_path = EVALUATION_DIR / "validation_comparison.csv"
    report_path = EVALUATION_DIR / "evaluation_summary.md"

    metric_fields = [
        "character_count",
        "word_count",
        "sentence_count",
        "unique_word_ratio",
        "repeated_trigram_ratio",
        "repeated_sentence_count",
    ]

    enriched = []

    for row in generations:
        metrics = text_metrics(row["generated_text"])
        enriched.append({**row, **metrics})

    with per_sample_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(enriched[0].keys())
        )
        writer.writeheader()
        writer.writerows(enriched)

    model_summary = {}

    for model in EXPERIMENTS:
        rows = [r for r in enriched if r["model"] == model]

        if not rows:
            raise ValueError(f"No generation rows found for {model}")

        model_summary[model] = {
            "sample_count": len(rows),
            **{
                f"mean_{field}": round(
                    mean([float(r[field]) for r in rows]), 4
                )
                for field in metric_fields
            },
            "batch_size": EXPERIMENTS[model]["batch_size"],
        }

    validation = {
        model: read_validation_log(info["run"])
        for model, info in EXPERIMENTS.items()
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generation_summary": model_summary,
                "validation_summary": validation,
                "interpretation_note": (
                    "Text statistics are diagnostic proxies, not "
                    "validated measures of story quality. Review "
                    "samples manually. Batch sizes differed between "
                    "the two training runs."
                ),
            },
            f,
            indent=2,
        )

    with validation_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = ["model", *validation["25k"].keys()]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for model, record in validation.items():
            writer.writerow({"model": model, **record})

    report_lines = [
        "# Mini LLM Evaluation Summary",
        "",
        "## Generation statistics",
        "",
        "| Metric | 25k | 100k |",
        "|---|---:|---:|",
    ]

    display_names = {
        "sample_count": "Sample count",
        "mean_word_count": "Mean word count",
        "mean_sentence_count": "Mean sentence count",
        "mean_unique_word_ratio": "Mean unique-word ratio",
        "mean_repeated_trigram_ratio": "Mean repeated-trigram ratio",
        "mean_repeated_sentence_count": "Mean repeated sentences",
    }

    for key, label in display_names.items():
        report_lines.append(
            f"| {label} | {model_summary['25k'][key]:.4f} "
            f"| {model_summary['100k'][key]:.4f} |"
        )

    report_lines += [
        "",
        "## Validation results",
        "",
        "| Metric | 25k | 100k |",
        "|---|---:|---:|",
    ]

    for key, label in [
        ("best_validation_loss", "Best validation loss"),
        ("best_validation_perplexity", "Perplexity at best-loss step"),
        ("latest_validation_loss", "Latest validation loss"),
        ("latest_validation_perplexity", "Latest validation perplexity"),
    ]:
        a = validation["25k"].get(key)
        b = validation["100k"].get(key)

        report_lines.append(
            f"| {label} | {a if a is not None else 'N/A'} "
            f"| {b if b is not None else 'N/A'} |"
        )

    report_lines += [
        "",
        "## Limitations",
        "",
        "- Automatic text statistics do not establish semantic coherence.",
        "- Manually review the paired generations for fluency, logical "
        "consistency, repetition and relevance to the prompt.",
        "- Training batch sizes differed (288 versus 1280), so this is "
        "not a controlled dataset-size-only comparison.",
        "- Validation results are comparable only if the validation "
        "data, tokenizer and evaluation procedure match.",
        "- Generation runtime includes process startup and checkpoint "
        "loading; it is not a clean throughput benchmark.",
        "",
    ]

    report_path.write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("=" * 68)
    print("MINI LLM EVALUATION ANALYSIS COMPLETED")
    print("=" * 68)

    for model in EXPERIMENTS:
        print(f"\n{model} model")
        print(f"  Samples: {model_summary[model]['sample_count']}")
        print(
            "  Mean word count: "
            f"{model_summary[model]['mean_word_count']:.2f}"
        )
        print(
            "  Unique-word ratio: "
            f"{model_summary[model]['mean_unique_word_ratio']:.4f}"
        )
        print(
            "  Repeated-trigram ratio: "
            f"{model_summary[model]['mean_repeated_trigram_ratio']:.4f}"
        )
        print(
            "  Best validation loss: "
            f"{validation[model]['best_validation_loss']:.6f}"
        )

    print("\nSaved files:")
    print(f"  {per_sample_path}")
    print(f"  {summary_path}")
    print(f"  {validation_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
