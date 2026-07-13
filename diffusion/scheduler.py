"""
Defines the diffusion schedule: what fraction of tokens should still
be masked at a given step, following either a linear or cosine
schedule. This module knows nothing about tokens, models, or
sampling — only about the shape of the schedule over time.
"""

import math


class DiffusionScheduler:
    """
    Computes the target mask ratio for a given diffusion step.

    Generation starts at step 0 (fully masked) and proceeds to
    step `total_steps` (fully unmasked). At each intermediate step,
    `get_mask_ratio` tells the generator what fraction of the
    editable sequence should still be masked.
    """

    def __init__(self, total_steps: int = 50, schedule: str = "cosine"):
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if schedule not in ("cosine", "linear"):
            raise ValueError(f"Unknown schedule: {schedule!r}")

        self.total_steps = total_steps
        self.schedule = schedule

    def get_mask_ratio(self, step: int) -> float:

        if step < 0 or step > self.total_steps:
            raise ValueError(f"step must be in [0, {self.total_steps}], got {step}")

        t = step / self.total_steps  

        if self.schedule == "linear":
            ratio = 1.0 - t
        else:  
            ratio = math.cos((math.pi / 2) * t)

        return max(0.0, min(1.0, ratio))

    def get_num_tokens_to_mask(self, step: int, seq_len: int) -> int:
        return round(self.get_mask_ratio(step) * seq_len)

    def steps(self):
        return range(self.total_steps + 1)

    def __repr__(self):
        return f"DiffusionScheduler(total_steps={self.total_steps}, schedule={self.schedule!r})"