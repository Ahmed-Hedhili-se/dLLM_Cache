"""   Public-facing orchestrator that ties together TokenTracker and every
individual cache (embedding, hidden, attention, moe) into one object
the model / generator actually talks to. Mirrors the role
diffusion/inference.py plays for the diffusion package: a single
clean entry point that hides the machinery underneath.
"""

from typing import List, Optional

from .token_tracker import TokenTracker, StepDiff
from .embedding_cache import EmbeddingCache
from .hidden_cache import HiddenCache
from .attention_cache import AttentionCache
from .moe_cache import MoECache
from .selective_compute import build_compute_plan, ComputePlan


class CacheManager:
    """
    Usage:
        cache_manager = CacheManager(mask_token_id=0, num_layers=24, embed_fn=model.embed_tokens)

        for step, tokens in enumerate(diffusion_steps):
            diff = cache_manager.begin_step(step, tokens)
            for layer_index in range(num_layers):
                plan = cache_manager.plan_layer(diff, layer_index, kind="hidden")
                # run the model only on plan.recompute_positions, then:
                # cache_manager.hidden.set_layer(layer_index, hidden_states, plan.recompute_positions)
    """

    def __init__(self, mask_token_id: int, num_layers: int, embed_fn=None):
        self.mask_token_id = mask_token_id
        self.num_layers = num_layers

        self.token_tracker = TokenTracker(mask_token_id=mask_token_id)
        self.embedding = EmbeddingCache(embed_fn=embed_fn) if embed_fn is not None else None
        self.hidden = HiddenCache()
        self.attention = AttentionCache()
        self.moe = MoECache()

        self._last_diff: Optional[StepDiff] = None

    def reset(self):
        self.token_tracker.reset()
        if self.embedding is not None:
            self.embedding.clear()
        self.hidden.clear()
        self.attention.clear()
        self.moe.clear()
        self._last_diff = None

    def begin_step(self, step: int, tokens: List[int]) -> StepDiff:
        """
        Call once per diffusion step, before running the model. Updates
        the token tracker and invalidates cache entries for positions
        that changed, so stale data is never accidentally reused.
        """
        diff = self.token_tracker.update(step, tokens)

        if diff.revealed_positions:
            self.hidden.invalidate_positions(diff.revealed_positions, self.num_layers)
            self.attention.invalidate_positions(diff.revealed_positions, self.num_layers)
            self.moe.invalidate_positions(diff.revealed_positions, self.num_layers)

        self._last_diff = diff
        return diff

    def plan_layer(
        self, diff: StepDiff, layer_index: int,
          kind: str = "hidden", force_full_recompute: bool = False
    ) -> ComputePlan:
        cache = {"hidden": self.hidden, "attention": self.attention, "moe": self.moe}.get(kind)
        if cache is None:
            raise ValueError(f"Unknown cache kind: {kind!r}")

        return build_compute_plan(diff, layer_index, cache, force_full_recompute=force_full_recompute)


    def stats(self) -> dict:
        return {
            "embedding_entries": len(self.embedding) if self.embedding is not None else 0,
            "hidden_entries": len(self.hidden),
            "attention_entries": len(self.attention),
            "moe_entries": len(self.moe),
            "stable_ratio": self.token_tracker.stable_ratio(),
        }