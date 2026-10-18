
"""Automated tests for autoregressive text generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mini_llm.generation import generate_text
from mini_llm.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID


class FakeTokenizer:
    """Small UTF-8 byte tokenizer for isolated generation tests."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8")


class FakeModel(nn.Module):
    """Deterministic test model that emits predefined token logits."""

    def __init__(
        self,
        next_tokens: list[int],
        block_size: int = 8,
        alternatives: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(block_size=block_size)
        self.next_tokens = next_tokens
        self.alternatives = alternatives
        self.calls = 0
        self.seen_context_lengths: list[int] = []

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_context_lengths.append(input_ids.shape[1])

        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, 259),
            fill_value=-100.0,
            device=input_ids.device,
        )

        index = min(self.calls, len(self.next_tokens) - 1)
        next_token = self.next_tokens[index]
        self.calls += 1

        logits[:, -1, next_token] = 10.0

        if self.alternatives is not None:
            first, second = self.alternatives
            logits[:, -1, first] = 1.0
            logits[:, -1, second] = 0.9

        return {"logits": logits}


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def test_greedy_generation_continues_prompt(tokenizer: FakeTokenizer) -> None:
    model = FakeModel([ord("b"), ord("c")])

    result = generate_text(
        model,
        tokenizer,
        "a",
        max_new_tokens=2,
        temperature=0,
        device=torch.device("cpu"),
    )

    assert result == "abc"
    assert model.calls == 2


def test_generation_stops_at_eos(tokenizer: FakeTokenizer) -> None:
    model = FakeModel([ord("b"), EOS_TOKEN_ID, ord("c")])

    result = generate_text(
        model,
        tokenizer,
        "a",
        max_new_tokens=5,
        temperature=0,
        device=torch.device("cpu"),
    )

    assert result == "ab"
    assert model.calls == 2


def test_generation_is_reproducible_with_seed(
    tokenizer: FakeTokenizer,
) -> None:
    def run_once() -> str:
        model = FakeModel(
            [ord("x")],
            alternatives=(ord("x"), ord("y")),
        )
        return generate_text(
            model,
            tokenizer,
            "a",
            max_new_tokens=12,
            temperature=1.0,
            top_k=2,
            top_p=1.0,
            device=torch.device("cpu"),
            seed=123,
        )

    assert run_once() == run_once()


def test_context_never_exceeds_block_size(
    tokenizer: FakeTokenizer,
) -> None:
    model = FakeModel([ord("x")], block_size=3)

    generate_text(
        model,
        tokenizer,
        "abcdef",
        max_new_tokens=4,
        temperature=0,
        device=torch.device("cpu"),
    )

    assert model.seen_context_lengths
    assert max(model.seen_context_lengths) <= 3


def test_zero_new_tokens_returns_prompt(tokenizer: FakeTokenizer) -> None:
    model = FakeModel([ord("b")])

    result = generate_text(
        model,
        tokenizer,
        "hello",
        max_new_tokens=0,
        device=torch.device("cpu"),
    )

    assert result == "hello"
    assert model.calls == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": -1},
        {"temperature": -0.1},
        {"top_k": 0},
        {"top_p": 0},
        {"top_p": 1.1},
    ],
)
def test_invalid_generation_arguments_raise(
    tokenizer: FakeTokenizer,
    kwargs: dict,
) -> None:
    model = FakeModel([ord("b")])

    with pytest.raises(ValueError):
        generate_text(
            model,
            tokenizer,
            "a",
            device=torch.device("cpu"),
            **kwargs,
        )


def test_non_string_prompt_raises(tokenizer: FakeTokenizer) -> None:
    model = FakeModel([ord("b")])

    with pytest.raises(TypeError):
        generate_text(model, tokenizer, 123)  # type: ignore[arg-type]


def test_model_training_mode_is_restored(tokenizer: FakeTokenizer) -> None:
    model = FakeModel([ord("b")])
    model.train()

    generate_text(
        model,
        tokenizer,
        "a",
        max_new_tokens=1,
        temperature=0,
        device=torch.device("cpu"),
    )

    assert model.training is True
