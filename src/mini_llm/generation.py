
"""Autoregressive text generation for Mini LLM Laboratory."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from mini_llm.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID


@torch.inference_mode()
def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    stop_at_eos: bool = True,
) -> str:
    """Generate text from a prompt using autoregressive sampling.

    The model is evaluated one token at a time. Its input is cropped to the
    configured context length, so generation never exceeds the model's
    positional-embedding capacity.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1 when provided")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    # Encode the prompt as ordinary UTF-8 byte tokens.
    prompt_ids = list(tokenizer.encode(prompt))

    # The BOS token tells the model that this is the beginning of a sequence.
    if not prompt_ids or prompt_ids[0] != BOS_TOKEN_ID:
        prompt_ids.insert(0, BOS_TOKEN_ID)

    if not prompt_ids:
        raise ValueError("The prompt must produce at least one token")

    context_length = model.config.block_size
    input_ids = torch.tensor(
        [prompt_ids], dtype=torch.long, device=device
    )
    generated_ids = list(prompt_ids)

    # Use a dedicated random generator when a seed is supplied.
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            # Keep only the most recent context_length tokens.
            context = input_ids[:, -context_length:]

            output = model(context)

            if isinstance(output, dict):
                logits = output["logits"]
            elif isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output


            # Select the vocabulary scores for the next token.
            next_logits = logits[:, -1, :].float()

            # Temperature controls sampling randomness.
            if temperature == 0:
                next_token = torch.argmax(
                    next_logits, dim=-1, keepdim=True
                )
            else:
                next_logits = next_logits / temperature

                # Top-k: retain only the k highest-scoring tokens.
                if top_k is not None:
                    k = min(top_k, next_logits.size(-1))
                    threshold = torch.topk(
                        next_logits, k, dim=-1
                    ).values[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, float("-inf")
                    )

                # Top-p (nucleus sampling): retain the smallest
                # high-probability group whose cumulative probability
                # reaches p.
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(
                        sorted_probs, dim=-1
                    )

                    remove = cumulative_probs > top_p
                    # Keep the first token that crosses the threshold.
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False

                    remove_original_order = torch.zeros_like(remove)
                    remove_original_order.scatter_(
                        -1, sorted_indices, remove
                    )
                    next_logits = next_logits.masked_fill(
                        remove_original_order, float("-inf")
                    )

                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )

            token_id = int(next_token.item())

            if stop_at_eos and token_id == EOS_TOKEN_ID:
                break

            generated_ids.append(token_id)
            input_ids = torch.cat((input_ids, next_token), dim=1)

    finally:
        # Do not leave the model in evaluation mode if it was training before.
        model.train(was_training)

    # Byte tokens are IDs 0–255. BOS/EOS/PAD are control tokens and
    # must not be passed to the strict UTF-8 decoder.
    byte_ids = [token_id for token_id in generated_ids if 0 <= token_id <= 255]
    return tokenizer.decode(byte_ids)
