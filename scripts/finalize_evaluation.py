#!/usr/bin/env python3
"""Finalize automated Mini LLM evaluation reports without retraining models.

Reads reports/evaluation/generation_comparison.csv and, when available,
validation_comparison.csv. Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text.lower())


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1)))


def distinct_ratio(tokens: list[str], n: int) -> float:
    grams = ngram_counts(tokens, n)
    total = sum(grams.values())
    return len(grams) / total if total else 0.0


def repeated_ngram_occurrence_ratio(tokens: list[str], n: int) -> float:
    """Fraction of n-gram occurrences beyond the first occurrence."""
    grams = ngram_counts(tokens, n)
    total = sum(grams.values())
    repeats = sum(count - 1 for count in grams.values() if count > 1)
    return repeats / total if total else 0.0


def sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for chunk in chunks:
        normalized = " ".join(words(chunk))
        if normalized:
            result.append(normalized)
    return result


def terminal_punctuation(text: str) -> bool:
    # Ignore trailing whitespace and common closing quotation/bracket characters.
    cleaned = text.rstrip()
    cleaned = re.sub(r"""['"”’)\]}]+$""", "", cleaned).rstrip()
    return bool(cleaned) and cleaned[-1] in ".!?"


def find_column(headers: list[str], required_terms: tuple[str, ...]) -> str | None:
    lowered = {h: h.lower().replace(" ", "_") for h in headers}
    for original, normalized in lowered.items():
        if all(term in normalized for term in required_terms):
            return original
    return None


def normalize_model(value: str) -> str:
    value = (value or "").lower()
    if "100k" in value:
        return "100k"
    if "25k" in value:
        return "25k"
    return value.strip() or "unknown"


def load_validation(path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    if not rows:
        return {}
    headers = list(rows[0].keys())
    model_col = next((h for h in headers if "model" in h.lower() or "run_name" in h.lower()), None)
    loss_col = find_column(headers, ("loss",))
    ppl_col = next((h for h in headers if "perplex" in h.lower()), None)
    if not model_col:
        # Some comparison exports may use a checkpoint or experiment column.
        model_col = next((h for h in headers if "checkpoint" in h.lower() or "experiment" in h.lower()), None)
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        model = normalize_model(row.get(model_col, "")) if model_col else ""
        if model not in {"25k", "100k"}:
            joined = " ".join(str(v) for v in row.values()).lower()
            model = "100k" if "100k" in joined else ("25k" if "25k" in joined else "")
        if not model:
            continue
        values: dict[str, float] = {}
        for key, col in (("validation_loss", loss_col), ("perplexity", ppl_col)):
            if col:
                try:
                    values[key] = float(row[col])
                except (ValueError, TypeError):
                    pass
        if values:
            result[model] = values
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root (default: current directory)")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "reports" / "evaluation"
    generation_path = out_dir / "generation_comparison.csv"
    validation_path = out_dir / "validation_comparison.csv"

    rows = read_csv(generation_path)
    if not rows:
        raise SystemExit(f"No generation rows found in {generation_path}")
    required = {"model", "prompt", "generated_text"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"Generation CSV is missing required columns: {sorted(missing)}")

    metrics: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        model = normalize_model(row.get("model", ""))
        text = row.get("generated_text", "") or ""
        prompt = row.get("prompt", "") or ""
        continuation = text[len(prompt):].lstrip() if prompt and text.startswith(prompt) else text
        tokens = words(continuation)
        sent = sentences(continuation)
        sentence_counts = Counter(sent)
        repeated_sentence_count = sum(c - 1 for c in sentence_counts.values() if c > 1)
        try:
            limit = int(float(row.get("max_new_tokens", "") or 0))
        except ValueError:
            limit = 0
        # A byte-level tokenizer means generated characters roughly track generated tokens,
        # but UTF-8 and decoding mean this is only an approximation.
        near_limit = bool(limit and len(continuation.encode("utf-8", errors="replace")) >= 0.90 * limit)
        has_terminal = terminal_punctuation(continuation)
        last_word = tokens[-1] if tokens else ""
        item = {
            "sample_index": index,
            "model": model,
            "prompt": prompt,
            "max_new_tokens": limit,
            "continuation_characters": len(continuation),
            "continuation_words": len(tokens),
            "sentence_count_heuristic": len(sent),
            "mean_sentence_words_heuristic": round(mean([len(words(s)) for s in re.split(r"(?<=[.!?])\s+", continuation.strip()) if words(s)]), 4) if sent else 0.0,
            "distinct_1": round(distinct_ratio(tokens, 1), 6),
            "distinct_2": round(distinct_ratio(tokens, 2), 6),
            "distinct_3": round(distinct_ratio(tokens, 3), 6),
            "repeated_bigram_occurrence_ratio": round(repeated_ngram_occurrence_ratio(tokens, 2), 6),
            "repeated_trigram_occurrence_ratio": round(repeated_ngram_occurrence_ratio(tokens, 3), 6),
            "repeated_sentence_count": repeated_sentence_count,
            "has_terminal_punctuation_heuristic": has_terminal,
            "ends_without_terminal_punctuation_heuristic": not has_terminal,
            "near_generation_limit_heuristic": near_limit,
            "near_limit_without_terminal_punctuation_heuristic": near_limit and not has_terminal,
            "last_word": last_word,
            "generation_seconds": row.get("generation_seconds", ""),
        }
        metrics.append(item)
        by_model[model].append(item)

    validation = load_validation(validation_path)
    summary: dict[str, Any] = {
        "generation_csv": str(generation_path),
        "sample_count": len(rows),
        "samples_per_model": {m: len(v) for m, v in sorted(by_model.items())},
        "metrics_are_heuristics": True,
        "models": {},
        "comparison_limitations": [
            "Only the saved prompts and samples are measured; this is not a comprehensive estimate of all possible generations.",
            "Distinct-n measures surface n-gram diversity, not semantic quality or factuality.",
            "Terminal punctuation and near-limit flags are heuristics, not proof of sentence completeness or truncation.",
            "The 25k and 100k runs used different training batch sizes, so dataset size is not isolated as the only changed variable.",
        ],
    }
    for model in ("25k", "100k"):
        samples = by_model.get(model, [])
        if not samples:
            continue
        model_summary: dict[str, Any] = {"samples": len(samples)}
        numeric_fields = [
            "continuation_characters", "continuation_words", "sentence_count_heuristic",
            "mean_sentence_words_heuristic", "distinct_1", "distinct_2", "distinct_3",
            "repeated_bigram_occurrence_ratio", "repeated_trigram_occurrence_ratio",
            "repeated_sentence_count",
        ]
        for field in numeric_fields:
            model_summary[field] = round(mean(float(s[field]) for s in samples), 6)
        model_summary["terminal_punctuation_fraction"] = round(mean(float(s["has_terminal_punctuation_heuristic"]) for s in samples), 4)
        model_summary["near_generation_limit_fraction"] = round(mean(float(s["near_generation_limit_heuristic"]) for s in samples), 4)
        model_summary["near_limit_without_terminal_punctuation_fraction"] = round(mean(float(s["near_limit_without_terminal_punctuation_heuristic"]) for s in samples), 4)
        model_summary.update(validation.get(model, {}))
        summary["models"][model] = model_summary

    metrics_csv = out_dir / "final_evaluation_metrics.csv"
    summary_json = out_dir / "final_evaluation_summary.json"
    report_md = out_dir / "final_evaluation_report.md"
    write_csv(metrics_csv, metrics)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Mini LLM Final Automated Evaluation Report",
        "",
        f"Generation samples analyzed: **{len(rows)}**",
        "",
        "## Model comparison",
        "",
        "| Metric | 25k | 100k |",
        "|---|---:|---:|",
    ]
    display = [
        ("Samples", "samples", 0),
        ("Mean continuation words", "continuation_words", 2),
        ("Mean continuation characters", "continuation_characters", 1),
        ("Mean sentence words (heuristic)", "mean_sentence_words_heuristic", 2),
        ("Distinct-1", "distinct_1", 4),
        ("Distinct-2", "distinct_2", 4),
        ("Distinct-3", "distinct_3", 4),
        ("Repeated bigram occurrence ratio", "repeated_bigram_occurrence_ratio", 4),
        ("Repeated trigram occurrence ratio", "repeated_trigram_occurrence_ratio", 4),
        ("Repeated sentence count per sample", "repeated_sentence_count", 3),
        ("Terminal punctuation fraction (heuristic)", "terminal_punctuation_fraction", 3),
        ("Near generation limit fraction (heuristic)", "near_generation_limit_fraction", 3),
        ("Near limit and no terminal punctuation (heuristic)", "near_limit_without_terminal_punctuation_fraction", 3),
        ("Best validation loss", "validation_loss", 6),
        ("Validation perplexity", "perplexity", 6),
    ]
    for label, key, digits in display:
        vals = []
        for model in ("25k", "100k"):
            value = summary["models"].get(model, {}).get(key)
            vals.append("N/A" if value is None else (str(int(value)) if digits == 0 else f"{value:.{digits}f}"))
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")

    lines += [
        "",
        "## Interpretation guidance",
        "",
        "- Lower validation loss and perplexity indicate better next-token prediction on the same validation data; they do not guarantee better stories.",
        "- Higher distinct-n indicates greater surface-level n-gram diversity. It does not measure coherence or meaning.",
        "- Repetition ratios count repeated n-gram occurrences beyond their first occurrence; lower is less repetitive by this definition.",
        "- Terminal punctuation is checked after stripping common closing quotation/bracket characters.",
        "- The near-limit flag compares the UTF-8 byte length of the continuation with 90% of `max_new_tokens`. For a byte-level tokenizer this is a useful warning signal, but it is not an exact token count.",
        "- The completion indicators are heuristics. Inspect individual rows in `final_evaluation_metrics.csv` for outliers.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["comparison_limitations"])
    lines += [
        "",
        "## Files produced",
        "",
        f"- Per-sample diagnostics: `{metrics_csv.relative_to(root)}`",
        f"- Machine-readable summary: `{summary_json.relative_to(root)}`",
        f"- This report: `{report_md.relative_to(root)}`",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print("=" * 68)
    print("FINAL AUTOMATED EVALUATION REPORT CREATED")
    print("=" * 68)
    print(f"Generation rows: {len(rows)}")
    print(f"Samples per model: {summary['samples_per_model']}")
    for model, values in summary["models"].items():
        print(
            f"{model}: mean words={values['continuation_words']:.2f}, "
            f"distinct-2={values['distinct_2']:.4f}, "
            f"terminal punctuation={values['terminal_punctuation_fraction']:.2f}, "
            f"near limit={values['near_generation_limit_fraction']:.2f}"
        )
    print(f"Report: {report_md}")
    print(f"Per-sample CSV: {metrics_csv}")
    print(f"JSON summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
