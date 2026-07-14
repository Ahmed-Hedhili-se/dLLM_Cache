"""   Caches attention key/value projections, keyed by (layer_index, position).

Unlike a standard autoregressive KV-cache, diffusion attention is
usually bidirectional — every token attends to every other token, so
in principle a single changed token could affect the attention output
everywhere. This cache implements the common practical approximation
used in diffusion-LM caching schemes: K/V projections for STABLE
positions are reused as-is, and only the K/V for REVEALED (dirty)
positions are recomputed each step. Recombining reused and freshly
computed K/V into the actual attention output is the model/generator's
job, not this cache's — this file only stores and retrieves.
"""

from typing import Dict, List, Tuple
from .base_cache import BaseCache


class AttentionCache(BaseCache):
    def cache_kind(self) -> str:
        return "attention_kv"

    def set_kv(self, layer_index: int, position: int, key, value) -> None:
        self.set((layer_index, position), (key.clone(), value.clone()))

    def get_kv(self, layer_index: int, position: int):
        return self.get((layer_index, position))


    def gather_kv(self, layer_index: int, positions: List[int]) -> Dict[int, Tuple]:
        result = {}
        for pos in positions:
            kv = self.get_kv(layer_index, pos)
            if kv is not None:
                result[pos] = kv
        return result

    def invalidate_positions(self, positions: List[int], num_layers: int) -> None:
        keys_to_drop = [(layer, pos) for layer in range(num_layers) for pos in positions]
        self.invalidate(keys_to_drop)