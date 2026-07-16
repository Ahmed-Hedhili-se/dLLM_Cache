import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
from .utils import H, NE, TOPK
from .expert import ExpertMLP
from cache.selective_compute import gather_tokens, scatter_tokens

class MoEBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, NE, bias=False)
        self.experts = nn.ModuleList([ExpertMLP() for _ in range(NE)])

    def forward(
        self, 
        x: torch.Tensor,
        cache: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        k_step: Optional[int] = None,
        prompt_len: int = 0,
        update_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        
        if cache is None or cache_manager is None or k_step is None:
            return self._compute_moe(x)

        is_initial = cache_manager.is_initial_step(k_step)
        is_prompt_up = cache_manager.is_prompt_update(k_step)
        is_resp_full = cache_manager.is_response_full_update(k_step)

        out_full = torch.zeros_like(x)

        if is_initial or is_prompt_up:
            out_p = self._compute_moe(x[:, :prompt_len])
            out_full[:, :prompt_len] = out_p
            cache.update_prompt(out_p, prompt_len)
        else:
            out_full[:, :prompt_len] = cache.get_prompt()

       
        if is_initial or is_resp_full:
            out_r = self._compute_moe(x[:, prompt_len:])
            out_full[:, prompt_len:] = out_r
            cache.update_response(out_r)
        else:
            if update_indices is None or update_indices.shape[1] == 0:
                out_full[:, prompt_len:] = cache.get_response()
            else:
                x_partial = gather_tokens(x, update_indices)
                out_r_partial = self._compute_moe(x_partial)
                
                indices_resp = update_indices - prompt_len
                cache.update_response_partial(out_r_partial, indices_resp)
                out_full[:, prompt_len:] = cache.get_response()
                
        return out_full

    def _compute_moe(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to run the MoE calculation for a given input tensor x."""
        B, T, _ = x.shape
        x_flat = x.view(B * T, H)
        routing_weights = F.softmax(self.gate(x_flat), dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, TOPK, dim=-1)
        routing_weights = routing_weights.to(x.dtype)
        
        out = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=NE).permute(2, 1, 0)
        
        for expert_idx in range(NE):
            idx, top_x = torch.where(expert_mask[expert_idx])
            h = self.experts[expert_idx](x_flat[top_x]) * routing_weights[top_x, idx, None]
            out.index_add_(0, top_x, h.to(x.dtype))
            
        return out.view(B, T, H)