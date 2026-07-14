from typing import Optional, Any
import torch


class BaseCache:
    """
    Base class for caching intermediate features in dLLM-Cache.
    Features are partitioned into Prompt Cache (C_p) and Response Cache (C_r).
    """

    def __init__(self):
        self.prompt_cache: Optional[torch.Tensor] = None
        self.response_cache: Optional[torch.Tensor] = None
        self.prompt_len: int = 0

    def update_prompt(self, features: torch.Tensor, prompt_len: int):
        self.prompt_cache = features.clone().detach() if features is not None else None
        self.prompt_len = prompt_len


    def update_response(self, features: torch.Tensor):
        self.response_cache = features.clone().detach() if features is not None else None


    def update_response_partial(self, partial_features: torch.Tensor, update_indices: torch.Tensor):
        raise NotImplementedError("Partial update is specific to the tensor shape and logic of the subclass.")

    def get_prompt(self) -> Optional[torch.Tensor]:
        return self.prompt_cache

    def get_response(self) -> Optional[torch.Tensor]:
        return self.response_cache

    def get_full(self) -> Optional[torch.Tensor]:
        if self.prompt_cache is not None and self.response_cache is not None:
            return torch.cat([self.prompt_cache, self.response_cache], dim=1) # Assuming dim=1 is sequence length
        return None

    def reset(self):
        self.prompt_cache = None
        self.response_cache = None
        self.prompt_len = 0