"""Text generation service coordinating tokenization, inference, and truncation."""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from mini_llm.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID, ByteTokenizer

from backend.app.exceptions import GenerationExecutionError
from backend.app.schemas.generation import GenerationRequest, GenerationResponse
from backend.app.services.model_registry import LoadedModelEntry, ModelRegistry

logger = logging.getLogger("mini_llm.generator")


class TextGeneratorService:
    """Service handling text completion with fine-grained truncation control."""

    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        self.registry = registry or ModelRegistry.get_instance()
        self.tokenizer = self.registry.tokenizer

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Execute text generation from the requested model and apply constraints."""
        start_time = time.perf_counter()

        entry: LoadedModelEntry = self.registry.get_model_entry(request.model_id)
        model = entry.model
        device = self.registry.device

        # Encode input prompt
        try:
            prompt_byte_ids = list(self.tokenizer.encode(request.prompt))
        except Exception as exc:
            raise GenerationExecutionError(f"Failed to tokenize prompt: {exc}") from exc

        prompt_token_count = len(prompt_byte_ids)

        # Execute autoregressive generation
        try:
            raw_generated_ids, hit_token_limit = self._autoregressive_sample(
                model=model,
                prompt_ids=prompt_byte_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                seed=request.seed,
                stop_at_eos=request.stop_at_eos,
                device=device,
            )
        except Exception as exc:
            logger.exception("Error in autoregressive sampling: %s", exc)
            raise GenerationExecutionError(str(exc)) from exc

        # Decode newly generated tokens (excluding prompt)
        continuation_bytes = bytearray([tid for tid in raw_generated_ids if 0 <= tid <= 255])
        completion_token_count = len(continuation_bytes)

        # Decode safely into UTF-8 text
        completion_text = continuation_bytes.decode("utf-8", errors="replace")

        # Handle character-level truncation if specified
        truncated = hit_token_limit
        if request.max_characters is not None and len(completion_text) > request.max_characters:
            completion_text = completion_text[: request.max_characters]
            truncated = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        full_text = request.prompt + completion_text

        return GenerationResponse(
            model_id=request.model_id,
            prompt=request.prompt,
            completion=completion_text,
            full_text=full_text,
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
            total_tokens=prompt_token_count + completion_token_count,
            character_count=len(completion_text),
            truncated=truncated,
            inference_duration_ms=round(elapsed_ms, 2),
        )

    @torch.inference_mode()
    def _autoregressive_sample(
        self,
        model: torch.nn.Module,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
        seed: Optional[int],
        stop_at_eos: bool,
        device: torch.device,
    ) -> Tuple[list[int], bool]:
        """Perform step-by-step token generation.
        
        Returns:
            Tuple of (list_of_newly_generated_token_ids, boolean_indicating_token_limit_reached)
        """
        # Prepare prompt tensor with BOS token if absent
        input_ids_list = list(prompt_ids)
        if not input_ids_list or input_ids_list[0] != BOS_TOKEN_ID:
            input_ids_list.insert(0, BOS_TOKEN_ID)

        context_length = model.config.block_size
        input_tensor = torch.tensor([input_ids_list], dtype=torch.long, device=device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        was_training = model.training
        model.eval()

        new_generated_tokens: list[int] = []
        token_limit_reached = False

        try:
            for step in range(max_new_tokens):
                # Crop context to model's maximum block size
                context = input_tensor[:, -context_length:]
                output = model(context)

                if isinstance(output, dict):
                    logits = output["logits"]
                elif isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output

                next_logits = logits[:, -1, :].float()

                if temperature == 0.0:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature

                    if top_k is not None and top_k > 0:
                        k = min(top_k, next_logits.size(-1))
                        threshold = torch.topk(next_logits, k, dim=-1).values[:, -1].unsqueeze(-1)
                        next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                        sorted_probs = F.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                        remove = cumulative_probs > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False

                        remove_original_order = torch.zeros_like(remove)
                        remove_original_order.scatter_(-1, sorted_indices, remove)
                        next_logits = next_logits.masked_fill(remove_original_order, float("-inf"))

                    probabilities = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)

                token_id = int(next_token.item())

                if stop_at_eos and token_id == EOS_TOKEN_ID:
                    # Stopped naturally via EOS token
                    break

                new_generated_tokens.append(token_id)
                input_tensor = torch.cat((input_tensor, next_token), dim=1)

                if len(new_generated_tokens) >= max_new_tokens:
                    token_limit_reached = True
                    break

        finally:
            model.train(was_training)

        return new_generated_tokens, token_limit_reached
