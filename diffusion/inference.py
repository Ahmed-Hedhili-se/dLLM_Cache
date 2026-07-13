"""
Public API for diffusion-based generation. Wraps tokenization,
the DiffusionGenerator loop, and detokenization behind a single
`generate()` call.
"""

from typing import Optional

from .scheduler import DiffusionScheduler
from .generator import DiffusionGenerator


class DiffusionInference:

    def __init__(
        self,
        model,
        tokenizer=None,
        total_steps: int = 50,
        schedule: str = "cosine",
        mask_token_id: Optional[int] = None,
        sampling_strategy: str = "greedy",
        sampling_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if mask_token_id is None:
            mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "mask_token_id must be provided explicitly, or the tokenizer "
                "must expose a mask_token_id attribute."
            )
        self.mask_token_id = mask_token_id

        self.scheduler = DiffusionScheduler(total_steps=total_steps, schedule=schedule)
        self.generator = DiffusionGenerator(
            model_fn=self._model_fn,
            scheduler=self.scheduler,
            mask_token_id=self.mask_token_id,
            sampling_strategy=sampling_strategy,
            sampling_kwargs=sampling_kwargs,
        )

    def _model_fn(self, tokens: list):
        import torch

        input_ids = torch.tensor(tokens).unsqueeze(0) 
        with torch.no_grad():
            output = self.model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
        return logits[0]  # [seq_len, vocab_size]

    def generate(self, prompt, gen_length: int = 64):
        """
        Generates `gen_length` new tokens continuing from `prompt`.

        prompt: raw text (requires a tokenizer) or a pre-tokenized
                list of token ids.

        Returns decoded text if a tokenizer is set, otherwise a
        list of token ids.
        """
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("A tokenizer is required to accept text prompts")
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            prompt_ids = list(prompt)

        protected_positions = list(range(len(prompt_ids)))
        full_sequence = prompt_ids + [self.mask_token_id] * gen_length

        output_ids = self.generator.run(full_sequence, protected_positions=protected_positions)

        if self.tokenizer is not None:
            return self.tokenizer.decode(output_ids)
        return output_ids