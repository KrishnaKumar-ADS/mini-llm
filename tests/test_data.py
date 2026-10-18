
import pytest
import torch

from mini_llm.data import (
    DOCUMENT_SEPARATOR,
    LanguageModelDataset,
    create_dataloader,
    read_stories,
)
from mini_llm.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    ByteTokenizer,
)


@pytest.fixture
def tokenizer():
    return ByteTokenizer()


def write_stories(path, stories):
    path.write_text(
        DOCUMENT_SEPARATOR.join(stories),
        encoding="utf-8",
    )
    return path


def test_read_stories_preserves_story_boundaries(tmp_path):
    path = write_stories(
        tmp_path / "stories.txt",
        ["First story.", "Second story.", "Third story."],
    )

    assert read_stories(path) == [
        "First story.",
        "Second story.",
        "Third story.",
    ]


def test_dataset_returns_fixed_length_shifted_sequences(
    tmp_path, tokenizer
):
    path = write_stories(tmp_path / "stories.txt", ["ABC"])
    dataset = LanguageModelDataset(path, tokenizer, block_size=4)

    x, y = dataset[0]

    assert x.shape == (4,)
    assert y.shape == (4,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # BOS, A, B, C, EOS: each input token predicts its successor.
    assert x.tolist() == [BOS_TOKEN_ID, ord("A"), ord("B"), ord("C")]
    assert y.tolist() == [ord("A"), ord("B"), ord("C"), EOS_TOKEN_ID]


def test_stories_never_share_a_training_window(tmp_path, tokenizer):
    path = write_stories(tmp_path / "stories.txt", ["A", "B"])
    dataset = LanguageModelDataset(path, tokenizer, block_size=8)

    assert len(dataset) == 2

    x_first, y_first = dataset[0]
    x_second, y_second = dataset[1]

    assert x_first[0].item() == BOS_TOKEN_ID
    assert x_first[1].item() == ord("A")
    assert y_first[0].item() == ord("A")
    assert y_first[1].item() == EOS_TOKEN_ID

    assert x_second[0].item() == BOS_TOKEN_ID
    assert x_second[1].item() == ord("B")
    assert y_second[0].item() == ord("B")
    assert y_second[1].item() == EOS_TOKEN_ID

    # Padding is present only after each story's real targets.
    assert torch.all(y_first[2:] == PAD_TOKEN_ID)
    assert torch.all(y_second[2:] == PAD_TOKEN_ID)


def test_long_story_creates_multiple_windows(tmp_path, tokenizer):
    story = "A" * 20
    path = write_stories(tmp_path / "stories.txt", [story])

    dataset = LanguageModelDataset(path, tokenizer, block_size=8)

    assert len(dataset) > 1

    for index in range(len(dataset)):
        x, y = dataset[index]
        assert x.shape == (8,)
        assert y.shape == (8,)


def test_dataloader_batches_examples(tmp_path, tokenizer):
    path = write_stories(
        tmp_path / "stories.txt",
        ["First story.", "Second story.", "Third story."],
    )
    dataset = LanguageModelDataset(path, tokenizer, block_size=16)

    loader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    x, y = next(iter(loader))

    assert x.shape == (2, 16)
    assert y.shape == (2, 16)


def test_missing_file_raises_error(tmp_path, tokenizer):
    path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        LanguageModelDataset(path, tokenizer, block_size=8)


def test_empty_dataset_raises_error(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No stories found"):
        read_stories(path)


@pytest.mark.parametrize("block_size", [0, -1])
def test_invalid_block_size_is_rejected(
    tmp_path, tokenizer, block_size
):
    path = write_stories(tmp_path / "stories.txt", ["A"])

    with pytest.raises(ValueError, match="block_size"):
        LanguageModelDataset(path, tokenizer, block_size)
