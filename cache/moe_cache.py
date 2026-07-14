import torch
from .base_cache import BaseCache
from .selective_compute import scatter_tokens

class MoECache(BaseCache):
    def update_response_partial(self, partial_features: torch.Tensor, indices: torch.Tensor):
        if self.response_cache is None:
            raise RuntimeError("Cannot perform partial update before a full refresh has initialized the cache.")
            
        self.response_cache = scatter_tokens(self.response_cache, partial_features, indices)