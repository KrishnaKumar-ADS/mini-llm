
"""Prepare reproducible TinyStories datasets for controlled experiments."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SUBSETS_DIR = DATA_DIR / "subsets"

DATASET_ID = "roneneldan/TinyStories"
DATASET_LICENSE = "cdla-sharing-1.0"

TRAIN_POOL_SIZE = 250_000
TRAIN_SUBSET_SIZES = (25_000, 100_000, 250_000)
VALIDATION_SIZE = 5_000
TEST_SIZE = 5_000

SEED = 42
VALIDATION_SEED = 43
TEST_SEED = 44
SHUFFLE_BUFFER_SIZE = 50_000

DOCUMENT_SEPARATOR = "\n\n<|document_boundary|>\n\n"

CORPUS_PATH = RAW_DIR / "corpus.txt"
TRAIN_PATH = DATA_DIR / "train.txt"
VALIDATION_PATH = DATA_DIR / "validation.txt"
TEST_PATH = DATA_DIR / "test.txt"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def story_hash(story: str) -> str:
    """Calculate a stable fingerprint for a normalized story."""
    return hashlib.sha256(story.encode("utf-8")).hexdigest()


def clean_story(value: Any) -> str | None:
    """Normalize a story while preserving its wording."""
    if not isinstance(value, str):
        return None

    story = value.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not story:
        return None

    return story


def write_stories(path: Path, stories: list[str]) -> None:
    """Write stories with an explicit document boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    content = DOCUMENT_SEPARATOR.join(stories) + "\n"
    path.write_text(content, encoding="utf-8", newline="\n")


def collect_unique_stories(
    split_name: str,
    target_count: int,
    seed: int,
    excluded_hashes: set[str] | None = None,
) -> tuple[list[str], set[str], dict[str, int]]:
    """Collect a deterministic, version-pinned stream sample."""

    excluded = set(excluded_hashes or ())
    seen = set(excluded)
    stories: list[str] = []

    stats = {
        "records_scanned": 0,
        "invalid_records": 0,
        "duplicate_or_excluded_records": 0,
    }

    print(f"\nReading upstream split: {split_name}")
    print(f"Target: {target_count:,} unique stories")

    stream = load_dataset(
        DATASET_ID,
        split=split_name,
        streaming=True,
        revision=DATASET_REVISION,
    )

    stream = stream.shuffle(
        seed=seed,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )

    for record in stream:
        stats["records_scanned"] += 1
        story = clean_story(record.get("text"))

        if story is None:
            stats["invalid_records"] += 1
            continue

        fingerprint = story_hash(story)

        if fingerprint in seen:
            stats["duplicate_or_excluded_records"] += 1
            continue

        stories.append(story)
        seen.add(fingerprint)

        if len(stories) % 25_000 == 0:
            print(
                f"Collected {len(stories):,}/{target_count:,} "
                f"from {split_name}"
            )

        if len(stories) == target_count:
            break

    if len(stories) != target_count:
        raise RuntimeError(
            f"Could not collect {target_count:,} unique stories "
            f"from upstream split '{split_name}'. "
            f"Collected {len(stories):,}. "
            "Check the source split, revision, and connection."
        )

    print(f"Collected: {len(stories):,}")
    print(f"Records scanned: {stats['records_scanned']:,}")
    print(
        "Invalid or duplicate records skipped: "
        f"{stats['invalid_records'] + stats['duplicate_or_excluded_records']:,}"
    )

    return stories, {story_hash(s) for s in stories}, stats


def describe_stories(stories: list[str]) -> dict[str, int]:
    """Summarize a set of stories."""
    return {
        "story_count": len(stories),
        "character_count": sum(len(s) for s in stories),
        "word_count": sum(len(s.split()) for s in stories),
    }


def describe_file(path: Path) -> dict[str, Any]:
    """Record the identity and size of a generated file."""
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve the upstream commit once and use it for every split.
    print(f"Resolving dataset revision: {DATASET_ID}")
    dataset_info = HfApi().dataset_info(DATASET_ID)
    global DATASET_REVISION
    DATASET_REVISION = dataset_info.sha

    card_data = getattr(dataset_info, "card_data", None) or {}
    if hasattr(card_data, "to_dict"):
        card_data = card_data.to_dict()

    upstream_license = (
        card_data.get("license")
        if isinstance(card_data, dict)
        else None
    )

    if upstream_license and upstream_license != DATASET_LICENSE:
        raise RuntimeError(
            f"Unexpected dataset license: {upstream_license!r}. "
            "Review the dataset card before continuing."
        )

    print(f"Pinned revision: {DATASET_REVISION}")
    print(f"Dataset license: {DATASET_LICENSE}")

    # The training pool is sampled only from the upstream training split.
    train_stories, train_hashes, train_stats = collect_unique_stories(
        split_name="train",
        target_count=TRAIN_POOL_SIZE,
        seed=SEED,
    )

    # Use the upstream validation split for two disjoint evaluation sets.
    # Test stories are never included in the training pool.
    validation_stories, validation_hashes, validation_stats = (
        collect_unique_stories(
            split_name="validation",
            target_count=VALIDATION_SIZE,
            seed=VALIDATION_SEED,
            excluded_hashes=train_hashes,
        )
    )

    test_stories, test_hashes, test_stats = collect_unique_stories(
        split_name="validation",
        target_count=TEST_SIZE,
        seed=TEST_SEED,
        excluded_hashes=train_hashes | validation_hashes,
    )

    # Fail immediately if document-level leakage exists.
    assert train_hashes.isdisjoint(validation_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert validation_hashes.isdisjoint(test_hashes)

    # Store the full selected training pool and canonical split files.
    write_stories(CORPUS_PATH, train_stories)
    write_stories(TRAIN_PATH, train_stories)
    write_stories(VALIDATION_PATH, validation_stories)
    write_stories(TEST_PATH, test_stories)

    subset_paths: dict[str, Path] = {}

    # All training subsets are nested: smaller sets are prefixes of
    # the same shuffled training pool.
    for subset_size in TRAIN_SUBSET_SIZES:
        subset = train_stories[:subset_size]
        subset_path = SUBSETS_DIR / f"train_{subset_size // 1000}k.txt"
        write_stories(subset_path, subset)
        subset_paths[str(subset_size)] = subset_path

    all_paths = {
        "raw_corpus": CORPUS_PATH,
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "test": TEST_PATH,
        **{
            f"train_subset_{size}": path
            for size, path in subset_paths.items()
        },
    }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": upstream_license or DATASET_LICENSE,
            "training_source_split": "train",
            "evaluation_source_split": "validation",
            "source_url": (
                "https://huggingface.co/datasets/roneneldan/TinyStories"
            ),
        },
        "preparation": {
            "seed_train": SEED,
            "seed_validation": VALIDATION_SEED,
            "seed_test": TEST_SEED,
            "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
            "sampling_method": (
                "Version-pinned streaming shuffle with seeded buffer"
            ),
            "deduplication": "Exact normalized-story SHA-256 fingerprints",
            "document_separator": "<|document_boundary|>",
            "training_subset_policy": (
                "Nested prefixes of the same shuffled training pool"
            ),
            "test_policy": (
                "Disjoint sample from the upstream validation split; "
                "not a separately published official test split"
            ),
        },
        "counts": {
            "training_pool": describe_stories(train_stories),
            "validation": describe_stories(validation_stories),
            "test": describe_stories(test_stories),
            "training_subsets": {
                str(size): describe_stories(train_stories[:size])
                for size in TRAIN_SUBSET_SIZES
            },
        },
        "collection_statistics": {
            "train": train_stats,
            "validation": validation_stats,
            "test": test_stats,
        },
        "files": {
            name: describe_file(path)
            for name, path in all_paths.items()
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nDATA PREPARATION COMPLETE")
    print("=" * 55)

    for name, count in (
        ("Training pool", len(train_stories)),
        ("Validation", len(validation_stories)),
        ("Test", len(test_stories)),
    ):
        print(f"{name:16} {count:>10,} stories")

    for size, path in subset_paths.items():
        print(f"Subset {size:>6}  {path.relative_to(PROJECT_ROOT)}")

    print("\nLeakage checks passed.")
    print(f"Manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()