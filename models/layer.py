import torch
import torch.nn as nn
from typing import Optional, Any, Dict
from .utils import H
from .rmsnorm import RMSNorm
from .attention import Attention
from .moe import MoEBlock

class Layer(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(H)
        self.self_attn = Attention()
        self.post_attention_layernorm = RMSNorm(H)
        self.mlp = MoEBlock()

    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        layer_cache: Optional[Dict[str, Any]] = None,
        cache_manager: Optional[Any] = None,
        k_step: Optional[int] = None,
        prompt_len: int = 0,
        update_ratio: float = 0.25
    ) -> torch.Tensor:
        attn_cache = layer_cache.get('attn') if layer_cache is not None else None
        mlp_cache = layer_cache.get('mlp') if layer_cache is not None else None

        # 1. Attention forward (returns output and updated indices if partial)
        attn_out, update_indices = self.self_attn(
            self.input_layernorm(x), 
            cos, 
            sin,
            cache=attn_cache,
            cache_manager=cache_manager,
            k_step=k_step,
            prompt_len=prompt_len,
            update_ratio=update_ratio
        )
        x = x + attn_out

        # 2. MLP/MoE forward (reuses the computed updated indices for FFN computation)
        mlp_out = self.mlp(
            self.post_attention_layernorm(x),
            cache=mlp_cache,
            cache_manager=cache_manager,
            k_step=k_step,
            prompt_len=prompt_len,
            update_indices=update_indices
        )
        x = x + mlp_out
        
        return x