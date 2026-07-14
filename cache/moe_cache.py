"""  Caches Mixture-of-Experts routing decisions and expert outputs,
keyed by (layer_index, position). Specific to MoE architectures like
LLaDA-MoE: routing a token to an expert and running that expert is
one of the most expensive parts of a forward pass, so skipping it for
STABLE positions is a large potential win.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from .base_cache import BaseCache


@dataclass
class MoEEntry :
    expert_ids : Tuple[int, ...]
    routing_weights: Tuple[float, ...]
    output: object 


class MoECache( BaseCache ):
    """Keys are (layer_index, position) tuples. Values are MoEEntry instances."""

    def cache_kind(self) -> str:
        return  "MoE"




    def set_entry(
        self, layer_index: int, position: int,
        expert_ids: Sequence[int], routing_weights: Sequence[float], output,
    ) -> None:
        entry = MoEEntry(
            expert_ids=tuple(expert_ids),
            routing_weights=tuple(routing_weights),
            output=output.clone() if hasattr(output, "clone") else output,
        )
        self.set((layer_index   , position), entry)




    def get_entry(self, layer_index : int, position: int)-> MoEEntry:
        return self.get((layer_index, position))


    def gather_outputs(self, layer_index: int, positions: List[int]) -> Dict[int, object]:
        """Returns {position: cached_output} for whichever positions have a cached MoE result."""
        result = {}
        for pos in positions:
            entry = self.get_entry(layer_index, pos)
            if entry is not None:
                result[pos ] = entry.output
        return result

    def invalidate_positions(self, positions: List[int], num_layers: int) -> None:
        keys_to_drop = [(layer, pos) for layer in range(num_layers) for pos in positions]
        self.invalidate(keys_to_drop)