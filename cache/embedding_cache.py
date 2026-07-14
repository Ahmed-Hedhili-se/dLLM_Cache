import torch
from typing import Optional

"""
Caches the embeddings (and positional encodings) for prompt tokens.
Since prompt tokens never change across diffusion steps, they can be cached indefinitely.
Response token embeddings are not typically cached since embedding lookup is O(1) and very cheap, 
but the interface can support it if needed.

"""

class EmbeddingCache:
    def __init__(self):
        self.prompt_embeds: Optional[torch.Tensor] = None
    def update_prompt(self, embeds: torch.Tensor):
        self.prompt_embeds = embeds.clone().detach() if embeds is not None else None
    def get_prompt(self) -> Optional[torch.Tensor]:
        return self.prompt_embeds
    def reset(self)  :
        self.prompt_embeds = None