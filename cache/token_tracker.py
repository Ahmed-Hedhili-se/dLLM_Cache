"""
This module knows nothing about embeddings, hidden states, attention,
or MoE routing — it only tracks token IDs and their masked/stable/
revealed status over time. Every other cache file builds on the
answer this file provides."""

from dataclasses import dataclass ,field
from typing import List, Optional
from enum import Enum

class TokeniState(Enum):
    MASKED = "masked"
    STABLE = "stable"
    REVEALED = "revealed"

@dataclass
class StepDiff:
    step: int
    revealed_positions: List[int] = field(default_factory=list)
    stable_positions: List[int] = field(default_factory=list)
    masked_positions: List[int] = field(default_factory=list)

    @property
    def dirty_positions(self) ->  List[int]:
        return self.revealed_positions 
    
    def __repr__(self) -> str:
        return (
            f"StepDiff(step={self.step}, "
            f"revealed_positions={self.revealed_positions}, "
            f"stable_positions={self.stable_positions}, "
            f"masked_positions={self.masked_positions})"
        )
    

class TokenTracker : 
    """
    Compares the token sequence at each step against the previous step
    to classify every position as MASKED, REVEALED, or STABLE.
 
    Usage:
        tracker = TokenTracker(mask_token_id=0)
        for step, tokens in enumerate(diffusion_steps):
            diff = tracker.update(step, tokens)
            # diff.dirty_positions  -> must be recomputed
            # diff.stable_positions -> safe to reuse from cache
    """
    def __init__(self, mask_token_id: int):
        self.mask_token_id = mask_token_id
        self._previous_tokens: Optional[List[int]] = None
        self._history: List[StepDiff] = []

    def reset(self) : 
        self._previous_tokens = None
        self._history = []
    
    def update (self , step: int , tokens : List[int]) ->StepDiff: 
        revealed, stable , masked = [], [], []
        for pos , token_id in enumerate(tokens): 
            if token_id == self.mask_token_id: 
                masked.append(pos)
                continue
            was_masked_or_new =(
                self._previous_tokens is None
                or pos >= len(self._previous_tokens)
                or self._previous_tokens[pos] == self.mask_token_id
            )
            changed_value = (
                self._previous_tokens is not None
                and pos < len(self._previous_tokens)
                and self._previous_tokens[pos] != token_id
            )

            if was_masked_or_new and changed_value:
                revealed.append(pos)
            else :
                stable.append(pos)

        diff = StepDiff(
            step=step,
            revealed_positions=revealed,
            stable_positions=stable,
            masked_positions=masked
        )
        self._history.append(diff)
        self._previous_tokens = tokens
        return diff
    


    def history(self) ->  List[StepDiff]:
        return self._history
    

    def stable_ratio(self )-> float : 
        if not self._history:
            return 0.0
        latest = self._history[-1]
        unmasked = len(latest.stable_positions) + len(latest.revealed_positions)
        if unmasked == 0:
            return 0.0
        return len(latest.stable_positions) / unmasked

