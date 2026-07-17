"""
Caches the embeddings (and positional encodings) for prompt tokens.
Since prompt tokens never change across diffusion steps, they can be
cached indefinitely (refreshed only on prompt-refresh steps, same as
AttentionCache's prompt half and MoECache's prompt half).
 
Response token embeddings are not cached: embedding lookup is O(1) and
cheap enough that caching it isn't worthwhile, and unlike the prompt,
response tokens are still actively changing across steps.
 
Note: this is not one of the four feature types the paper caches -- it
is a safe, zero-risk bonus optimization (an embedding lookup is a pure
function of token id, so there is no approximation involved here,
unlike every other cache in this package).
"""

import torch
from typing import Optional


class EmbeddingCache:
    def __init__(self):
        self.prompt_embeds: Optional[torch.Tensor] = None
    def update_prompt(self, embeds: torch.Tensor):
        self.prompt_embeds = embeds.detach() if embeds is not None else None
    def get_prompt(self) -> Optional[torch.Tensor]:
        return self.prompt_embeds
    def reset(self)  :
        self.prompt_embeds = None