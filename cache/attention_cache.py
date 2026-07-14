import torch
from typing import Optional, Tuple
from .selective_compute import scatter_tokens

class AttentionCache:
    def __init__(self):
        self.k_p: Optional[torch.Tensor] = None
        self.v_p: Optional[torch.Tensor] = None
        self.attn_out_p: Optional[torch.Tensor] = None
        
        self.k_r: Optional[torch.Tensor] = None
        self.v_r: Optional[torch.Tensor] = None
        self.attn_out_r: Optional[torch.Tensor] = None
        
        self.prompt_len: int = 0



    def update_prompt( self, k : torch.Tensor, v: torch.Tensor, attn_out : torch.Tensor, prompt_len: int):
        self.k_p = k.clone().detach()    if     k is not None else None
        self.v_p = v.clone().detach()    if     v is not None else None
        self.attn_out_p = attn_out.clone().detach() if attn_out is not None else None
        self.prompt_len = prompt_len

    def update_response(self, k: torch.Tensor, v: torch.Tensor, attn_out: torch.Tensor):
        self.k_r = k.clone().detach() if k is not None else None
        self.v_r = v.clone().detach() if v is not None else None
        self.attn_out_r = attn_out.clone().detach() if attn_out is not None else None


    def update_response_partial(self, partial_k: torch.Tensor, new_v: torch.Tensor, partial_attn_out: torch.Tensor, indices: torch.Tensor):
        """
        Adaptive partial update. Overwrites the full V_r cache (Section A.5 of paper)
        but only scatters K and AttnOut for the selected tokens.
        """
        if self.k_r is None or self.attn_out_r is None:
            raise RuntimeError("Cannot perform partial update before a full refresh has initialized the cache.")
            
        self.v_r = new_v.clone().detach()
        self.k_r = scatter_tokens(self.k_r, partial_k, indices)
        self.attn_out_r = scatter_tokens(self.attn_out_r, partial_attn_out, indices)


    def get_full(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.k_p is None or self.k_r is None:
            return None, None, None
        k = torch.cat( [self.k_p, self.k_r], dim=1)
        v = torch.cat([self.v_p, self.v_r], dim=1)
        attn_out = torch.cat([self.attn_out_p, self.attn_out_r], dim=1)
        
        return k, v, attn_out


    def get_cached_v_response(self) -> Optional[torch.Tensor]:
        return self.v_r