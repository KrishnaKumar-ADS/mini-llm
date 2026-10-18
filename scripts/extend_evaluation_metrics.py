
from __future__ import annotations

import csv
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/evaluation/generation_comparison.csv"
OUT = ROOT / "reports/evaluation"
OUTPUT = OUT / "extended_generation_metrics.csv"
SUMMARY = OUT / "extended_metrics_summary.md"


def words(text: str) -> list[str]:
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())


def distinct_ratio(tokens: list[str], n: int) -> float:
    grams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]
    return len(set(grams)) / len(grams) if grams else 0.0


def analyze(text: str) -> dict:
    tokens = words(text)
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]
    normalized = [
        re.sub(r"\s+", " ", s.lower()).strip()
        for s in sentences
    ]

    sentence_counts = Counter(normalized)
    repeated_sentences = sum(
        count - 1 for count in sentence_counts.values()
    )

    sentence_lengths = [len(words(s)) for s in sentences]
    punctuation_ending = bool(
        text.rstrip() and text.rstrip()[-1] in ".!?"
    )

    return {
        "word_count": len(tokens),
        "character_count": len(text),
        "sentence_count": len(sentences),
        "mean_sentence_words": (
            statistics.mean(sentence_lengths)
            if sentence_lengths else 0.0
        ),
        "distinct_1": distinct_ratio(tokens, 1),
        "distinct_2": distinct_ratio(tokens, 2),
        "distinct_3": distinct_ratio(tokens, 3),
        "repeated_sentence_count": repeated_sentences,
        "has_terminal_punctuation": punctuation_ending,
        "ending_punctuation_warning": not punctuation_ending,
    }


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Generation results not found: {SOURCE}"
        )

    with SOURCE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Generation CSV is empty.")

    results = []
    for row in rows:
        results.append({
            "model": row["model"],
            "prompt": row["prompt"],
            **analyze(row["generated_text"]),
        })

    OUT.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    metrics = [
        "word_count",
        "mean_sentence_words",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repeated_sentence_count",
        "ending_punctuation_warning",
    ]

    summary = [
        "# Extended Mini LLM Generation Metrics",
        "",
        f"Samples analyzed: {len(results)}",
        "",
        "| Metric | 25k | 100k |",
        "|---|---:|---:|",
    ]

    for metric in metrics:
        values = {}
        for model in ("25k", "100k"):
            selected = [
                r[metric] for r in results
                if r["model"] == model
            ]
            values[model] = (
                sum(selected) / len(selected)
                if selected else 0.0
            )

        if metric == "ending_punctuation_warning":
            label = "Fraction without terminal punctuation (heuristic)"
        else:
            label = metric

        summary.append(
            f"| {label} | {values['25k']:.4f} "
            f"| {values['100k']:.4f} |"
        )

    summary += [
        "",
        "## Interpretation notes",
        "",
        "- Distinct-n measures lexical diversity of n-grams, not meaning.",
        "- Repeated-sentence count detects exact normalized repeats.",
        "- Missing terminal punctuation is only a warning; it does not "
        "prove that a generation was truncated.",
        "- Compare these diagnostics alongside validation loss and "
        "perplexity, not as a single overall quality score.",
        "- The models used different training batch sizes, so the "
        "comparison does not isolate dataset size alone.",
        "",
    ]

    SUMMARY.write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved per-sample metrics: {OUTPUT}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Rows analyzed: {len(results)}")


if __name__ == "__main__":
    main()
