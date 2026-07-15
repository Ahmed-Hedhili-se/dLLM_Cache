import torch
import torch.nn.functional as F
import math

"""
Implements the V-verify mechanism to select tokens for adaptive partial
updates (Eq. 7, Sec 3.2.2). Computes the cosine similarity between each
response token's current and cached Value vector, and selects the
tokens with the LOWEST similarity (i.e. the ones whose features drifted
the most since they were last cached) for full recomputation.
"""


class TokenTracker:

    def __init__(self, update_ratio: float = 0.25):
        self.update_ratio = update_ratio

    def verify( self, current_v: torch.Tensor, cached_v: torch.Tensor) -> torch.Tensor:
        B, T = current_v.shape[:2]

        v_c = current_v.reshape(B, T, -1)
        v_p = cached_v.reshape(B, T, -1)

        similarity =F.cosine_similarity(v_c, v_p, dim=-1)
        num_tokens_to_update= math.floor(self.update_ratio * T)

        if num_tokens_to_update == 0:
            return torch.empty((B, 0), dtype=torch.long, device=current_v.device )

        _, indices = torch.topk(similarity, k=num_tokens_to_update  , dim=-1 , largest=False)
        indices, _ = torch.sort(indices, dim=-1)
        return indices