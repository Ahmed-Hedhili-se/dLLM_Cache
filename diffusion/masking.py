"""
Everything related to masks: choosing which positions to mask,
applying the mask token, and restoring predicted tokens back into
the sequence. No scheduling logic, no model calls, no sampling —
just mask mechanics.
"""

import random
from typing import Optional, Sequence


def sample_mask_positions(
    seq_len: int,
    mask_ratio: float,
    rng: Optional[random.Random] = None,
    protected_positions: Optional[Sequence[int]] = None,
) -> list:
    

    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError("mask_ratio must be in [0, 1]")

    rng = rng or random
    protected = set(protected_positions or [])
    candidates = [i for i in range(seq_len) if i not in protected]

    num_to_mask = round(mask_ratio * len(candidates))
    if num_to_mask <= 0:
        return []

    return rng.sample(candidates, min(num_to_mask, len(candidates)))


def apply_mask(tokens: list, positions: Sequence[int], mask_token_id: int) -> list:
    
    out = list(tokens)
    for pos in positions:
        out[pos] = mask_token_id
    return out


def mask_tokens(
    tokens: list,
    mask_ratio: float,
    mask_token_id: int,
    rng: Optional[random.Random] = None,
    protected_positions: Optional[Sequence[int]] = None,
) -> tuple:
    
    positions = sample_mask_positions(
        len(tokens), mask_ratio, rng=rng, protected_positions=protected_positions
    )
    masked = apply_mask(tokens, positions, mask_token_id)
    return masked, positions


def restore_tokens(tokens: list, predictions: dict) -> list:
    out = list(tokens)
    for pos, token_id in predictions.items():
        out[pos] = token_id
    return out