"""
The core denoising loop: mask -> model -> sample -> update tokens,
repeated according to the schedule from scheduler.py. This is the
only file that orchestrates the other modules together.
"""

import random
from typing import Callable, Optional, Sequence

from .scheduler import DiffusionScheduler
from .masking import apply_mask, restore_tokens
from .sampling import sample, token_confidence

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class DiffusionGenerator:
    """
    Runs the iterative mask -> model -> sample -> update loop that
    turns a fully-masked sequence into a fully-denoised one.

    `model_fn` is any callable with signature:
        model_fn(tokens: list[int]) -> logits  (shape: [seq_len, vocab_size])
    This keeps the generator decoupled from any specific model class.
    """

    def __init__(
        self,
        model_fn: Callable,
        scheduler: Optional[DiffusionScheduler] = None,
        mask_token_id: int = 0,
        sampling_strategy: str = "greedy",
        sampling_kwargs: Optional[dict] = None,
        remask_by_confidence: bool = True,
        rng: Optional[random.Random] = None,
    ):
        self.model_fn = model_fn
        self.scheduler = scheduler or DiffusionScheduler()
        self.mask_token_id = mask_token_id
        self.sampling_strategy = sampling_strategy
        self.sampling_kwargs = sampling_kwargs or {}
        self.remask_by_confidence = remask_by_confidence
        self.rng = rng or random.Random()

    def run(
        self,
        tokens: list,
        protected_positions: Optional[Sequence[int]] = None,
    ) -> list:
        """
        Executes the full denoising loop over `tokens`.

        `tokens` should already contain any protected content
        (e.g. a prompt) unmasked; everything else is masked at
        step 0 and progressively revealed.

        protected_positions (e.g. prompt indices) are never masked
        or overwritten.
        """
        protected = set(protected_positions or [])
        editable = [i for i in range(len(tokens)) if i not in protected]

        current = apply_mask(list(tokens), editable, self.mask_token_id)

        for step in self.scheduler.steps():
            current = self._denoise_step(current, step, editable)

        return current

    def _denoise_step(self, tokens: list, step: int, editable: list) -> list:
        if not _HAS_TORCH:
            raise ImportError("generator.py requires torch to run the model")

        masked_positions = [i for i in editable if tokens[i] == self.mask_token_id]
        if not masked_positions:
            return tokens  

        target_ratio = self.scheduler.get_mask_ratio(step)
        num_should_be_unmasked = round((1 - target_ratio) * len(editable))
        already_unmasked = len(editable) - len(masked_positions)
        num_to_reveal = max(0, num_should_be_unmasked - already_unmasked)

        if num_to_reveal == 0:
            if step == 0:
                self.model_fn(tokens)
            return tokens

        logits = self.model_fn(tokens)  # [seq_len, vocab_size]
        predicted_ids = sample(logits, strategy=self.sampling_strategy, **self.sampling_kwargs)

        if self.remask_by_confidence:
            confidences = token_confidence(logits, predicted_ids)
            ranked = sorted(masked_positions, key=lambda i: confidences[i].item(), reverse=True)
            reveal_positions = ranked[:num_to_reveal]
        else:
            reveal_positions = self.rng.sample(
                masked_positions, min(num_to_reveal, len(masked_positions))
            )

        predictions = {pos: predicted_ids[pos].item() for pos in reveal_positions}
        return restore_tokens(tokens, predictions)