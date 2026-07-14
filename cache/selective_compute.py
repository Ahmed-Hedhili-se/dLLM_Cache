"""  The decision-making layer that ties token_tracker.py to the caches.
Given a StepDiff (from TokenTracker), this module decides exactly
which positions need a fresh forward pass at a given layer, and how
to merge cached results back in for the rest — without storing
anything itself.
"""

from dataclasses import dataclass
from typing import Dict, List
from .token_tracker import StepDiff


@dataclass
class ComputePlan:
    layer_index: int
    recompute_positions: List[int]
    reuse_positions: List[int]

    def __repr__(self):
        return (
            f"ComputePlan(layer={self.layer_index}, "
            f"recompute={len(self.recompute_positions)}, "
            f"reuse={len(self.reuse_positions)})"
        )


def build_compute_plan(
    diff: StepDiff, layer_index: int,
    cache, force_full_recompute: bool = False
) -> ComputePlan:
    """
    Decides which positions must be recomputed at `layer_index` this
    step, versus which can be pulled from `cache` (any object exposing
    get_layer(layer_index, positions) -> {position: value}, e.g.
    HiddenCache or AttentionCache).

    A position is only reused if BOTH:
      - it was STABLE this step (per the token tracker), and
      - a cached value actually exists for it at this layer.
    Anything revealed this step, or missing from cache, gets recomputed.
    """
    if force_full_recompute:
        all_positions = diff.stable_positions + diff.revealed_positions
        return ComputePlan(layer_index=layer_index, recompute_positions=all_positions, reuse_positions=[])

    cached = cache.get_layer(layer_index, diff.stable_positions)
    reuse_positions = list(cached.keys())
    stale_positions = [p for p in diff.stable_positions if p not in cached]

    recompute_positions = diff.revealed_positions + stale_positions

    return ComputePlan(
        layer_index=layer_index,
        recompute_positions=recompute_positions,
        reuse_positions=reuse_positions,
    )


def merge_computed_and_cached(
    computed: Dict[int, object], 
    cached: Dict[int, object], seq_len: int, fill=None
) -> list:
    merged = [fill] * seq_len
    for pos, value in cached.items():
        merged[pos] = value
    for pos, value in computed.items():
        merged[pos] = value  # freshly computed values take priority over cached ones
    return merged