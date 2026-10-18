
"""A deterministic UTF-8 byte tokenizer for the Mini LLM Laboratory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


BYTE_VOCAB_SIZE = 256

EOS_TOKEN_ID = 256
BOS_TOKEN_ID = 257
PAD_TOKEN_ID = 258

VOCAB_SIZE = 259

SPECIAL_TOKENS = {
    "EOS": EOS_TOKEN_ID,
    "BOS": BOS_TOKEN_ID,
    "PAD": PAD_TOKEN_ID,
}


class ByteTokenizer:
    """Encode UTF-8 bytes directly as token IDs."""

    vocab_size = VOCAB_SIZE
    eos_token_id = EOS_TOKEN_ID
    bos_token_id = BOS_TOKEN_ID
    pad_token_id = PAD_TOKEN_ID

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert text into byte IDs, optionally adding special tokens."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        token_ids = list(text.encode("utf-8"))

        if add_bos:
            token_ids.insert(0, BOS_TOKEN_ID)

        if add_eos:
            token_ids.append(EOS_TOKEN_ID)

        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert byte IDs back into text."""
        decoded_bytes = bytearray()
        output: list[str] = []

        def flush_bytes() -> None:
            if decoded_bytes:
                output.append(decoded_bytes.decode("utf-8", errors="strict"))
                decoded_bytes.clear()

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError(
                    f"Token IDs must be integers, got {type(token_id).__name__}"
                )

            if 0 <= token_id < BYTE_VOCAB_SIZE:
                decoded_bytes.append(token_id)
                continue

            if token_id in SPECIAL_TOKENS.values():
                flush_bytes()

                if not skip_special_tokens:
                    token_name = next(
                        name
                        for name, value in SPECIAL_TOKENS.items()
                        if value == token_id
                    )
                    output.append(f"<|{token_name.lower()}|>")

                continue

            raise ValueError(
                f"Invalid token ID {token_id}; expected an ID in "
                f"[0, {VOCAB_SIZE - 1}]"
            )

        flush_bytes()
        return "".join(output)

    def save(self, path: str | Path) -> None:
        """Save the tokenizer's format and special-token IDs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "tokenizer_type": "utf8_byte",
            "format_version": 1,
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
        }

        path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        """Load and validate tokenizer metadata."""
        path = Path(path)
        metadata = json.loads(path.read_text(encoding="utf-8"))

        if metadata.get("tokenizer_type") != "utf8_byte":
            raise ValueError("Unsupported tokenizer type")

        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported tokenizer format version")

        if metadata.get("vocab_size") != VOCAB_SIZE:
            raise ValueError("Tokenizer vocabulary size mismatch")

        if metadata.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("Tokenizer special-token IDs mismatch")

        return cls()
