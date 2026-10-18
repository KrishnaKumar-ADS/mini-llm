
"""Dataset utilities for next-token language-model training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from mini_llm.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    ByteTokenizer,
)

# Must match the separator written by scripts/prepare_data.py.
DOCUMENT_SEPARATOR = "\n\n<|document_boundary|>\n\n"


def read_stories(path: str | Path) -> list[str]:
    """Read a prepared text file and return its non-empty stories."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    text = path.read_text(encoding="utf-8")
    stories = [
        story.strip()
        for story in text.split(DOCUMENT_SEPARATOR)
        if story.strip()
    ]

    if not stories:
        raise ValueError(f"No stories found in dataset file: {path}")

    return stories


class LanguageModelDataset(Dataset):
    """Fixed-length input/target windows that never cross story boundaries.

    Each input contains up to `block_size` token IDs.
    Each target is shifted one position to predict the next token.

    Incomplete final windows are padded with PAD_TOKEN_ID. Training code
    should ignore PAD_TOKEN_ID when calculating the loss.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer,
        block_size: int,
    ) -> None:
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise TypeError("block_size must be an integer")

        if block_size < 1:
            raise ValueError("block_size must be at least 1")

        self.path = Path(path)
        self.block_size = block_size
        self.tokenizer = tokenizer

        stories = read_stories(self.path)

        # int16 is sufficient for token IDs 0–258 and uses less memory
        # than storing every token as a 64-bit integer.
        self._encoded_stories: list[torch.Tensor] = []
        self._examples: list[tuple[int, int]] = []

        for story_index, story in enumerate(stories):
            token_ids = tokenizer.encode(
                story,
                add_bos=True,
                add_eos=True,
            )

            tokens = torch.tensor(token_ids, dtype=torch.int16)
            self._encoded_stories.append(tokens)

            # Each window contains block_size + 1 tokens before padding.
            # Adjacent windows overlap by one token, preserving the
            # next-token prediction alignment.
            for start in range(0, len(token_ids) - 1, block_size):
                self._examples.append((story_index, start))

        if not self._examples:
            raise ValueError(
                f"No training windows could be created from: {self.path}"
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        story_index, start = self._examples[index]
        tokens = self._encoded_stories[story_index]

        window = tokens[start : start + self.block_size + 1].long()

        # Shift by one token: x predicts the next token in y.
        x = window[:-1]
        y = window[1:]

        padding_length = self.block_size - x.numel()

        if padding_length > 0:
            padding = torch.full(
                (padding_length,),
                PAD_TOKEN_ID,
                dtype=torch.long,
            )
            x = torch.cat((x, padding))
            y = torch.cat((y, padding))

        return x, y


def create_dataloader(
    dataset: LanguageModelDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Create a reproducibly seeded DataLoader."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def create_train_validation_loaders(
    train_path: str | Path,
    validation_path: str | Path,
    tokenizer: ByteTokenizer,
    *,
    block_size: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build separate training and validation loaders."""
    train_dataset = LanguageModelDataset(
        train_path, tokenizer, block_size
    )
    validation_dataset = LanguageModelDataset(
        validation_path, tokenizer, block_size
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )

    return train_loader, validation_loader
