
"""Prepare a blinded human evaluation sheet for Mini LLM."""

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/evaluation/generation_comparison.csv"
OUT = ROOT / "reports/evaluation"

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing generation results: {SOURCE}")

with SOURCE.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("No generation samples found.")

required = {"model", "prompt", "generated_text"}
if not required.issubset(rows[0]):
    raise ValueError(f"CSV must contain columns: {required}")

# Fixed shuffle seed makes the review order reproducible.
rng = random.Random(20261018)
rng.shuffle(rows)

review_rows = []
key_rows = []

for i, row in enumerate(rows, start=1):
    sample_id = f"S{i:03d}"

    review_rows.append({
        "sample_id": sample_id,
        "prompt": row["prompt"],
        "generated_text": row["generated_text"],
        "fluency_1_to_5": "",
        "coherence_1_to_5": "",
        "prompt_relevance_1_to_5": "",
        "low_repetition_1_to_5": "",
        "notes": "",
    })

    key_rows.append({
        "sample_id": sample_id,
        "model": row["model"],
        "prompt": row["prompt"],
    })

review_path = OUT / "human_review_sheet.csv"
key_path = OUT / "human_review_key.csv"

OUT.mkdir(parents=True, exist_ok=True)

with review_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=review_rows[0].keys())
    writer.writeheader()
    writer.writerows(review_rows)

with key_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=key_rows[0].keys())
    writer.writeheader()
    writer.writerows(key_rows)

print(f"Samples prepared: {len(review_rows)}")
print(f"Review sheet: {review_path}")
print(f"Model key (keep separate): {key_path}")
print("Score the review sheet before opening the model key.")
