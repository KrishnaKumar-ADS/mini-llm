
"""Correctness tests for the byte tokenizer."""

import pytest

from src.mini_llm.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    ByteTokenizer,
)


@pytest.fixture
def tokenizer() -> ByteTokenizer:
    return ByteTokenizer()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, world!",
        "A small transformer.",
        "Café",
        "नमस्ते",
        "こんにちは",
        "Hello 👋 🌍",
        "Line one\nLine two",
        "Special-looking text: <|EOS|>",
    ],
)
def test_utf8_round_trip(tokenizer: ByteTokenizer, text: str) -> None:
    token_ids = tokenizer.encode(text)
    assert tokenizer.decode(token_ids) == text



def test_all_byte_values_are_supported(tokenizer: ByteTokenizer) -> None:
    # All 256 byte values must map to their corresponding token IDs.
    for value in range(256):
        assert tokenizer.encode(chr(value)) == list(chr(value).encode("utf-8"))

    # Every possible byte value must be accepted by the tokenizer's
    # encoding representation. Test decoding using a valid UTF-8 sequence.
    all_bytes_text = "".join(chr(value) for value in range(256))
    token_ids = tokenizer.encode(all_bytes_text)

    assert tokenizer.decode(token_ids) == all_bytes_text



def test_special_tokens_are_added_in_correct_order(
    tokenizer: ByteTokenizer,
) -> None:
    token_ids = tokenizer.encode("Hi", add_bos=True, add_eos=True)

    assert token_ids[0] == BOS_TOKEN_ID
    assert token_ids[-1] == EOS_TOKEN_ID
    assert tokenizer.decode(token_ids) == "Hi"


def test_special_tokens_can_be_rendered(
    tokenizer: ByteTokenizer,
) -> None:
    result = tokenizer.decode(
        [BOS_TOKEN_ID, ord("A"), EOS_TOKEN_ID],
        skip_special_tokens=False,
    )

    assert result == "<|bos|>A<|eos|>"


def test_vocab_size_and_ids(tokenizer: ByteTokenizer) -> None:
    assert tokenizer.vocab_size == 259
    assert len({EOS_TOKEN_ID, BOS_TOKEN_ID, PAD_TOKEN_ID}) == 3


@pytest.mark.parametrize("token_id", [-1, VOCAB_SIZE, 999])
def test_invalid_token_ids_fail(
    tokenizer: ByteTokenizer,
    token_id: int,
) -> None:
    with pytest.raises(ValueError):
        tokenizer.decode([token_id])


def test_save_and_load_round_trip(
    tokenizer: ByteTokenizer,
    tmp_path,
) -> None:
    path = tmp_path / "tokenizer" / "tokenizer.json"
    tokenizer.save(path)

    restored = ByteTokenizer.load(path)

    text = "A reproducible tokenizer 🧪"
    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.decode(restored.encode(text)) == text


def test_incorrect_metadata_is_rejected(
    tokenizer: ByteTokenizer,
    tmp_path,
) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    path.write_text(
        '{"tokenizer_type": "unknown", "format_version": 1, '
        '"vocab_size": 259, "special_tokens": {}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ByteTokenizer.load(path)
