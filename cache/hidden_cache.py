import torch
from typing import Optional, Tuple
from .selective_compute import scatter_tokens
from .base_cache import BaseCache
"""
    Manages the cache for the FFN Output tensor for a single Transformer layer.
"""
class HiddenCache(BaseCache):
    
    
    def update_response_partial(self, partial_features: torch.Tensor, indices: torch.Tensor):
        if self.response_cache is None:
            raise RuntimeError("Cannot perform partial update before a full refresh has initialized the cache.")
        self.response_cache = scatter_tokens(self.response_cache, partial_features, indices)