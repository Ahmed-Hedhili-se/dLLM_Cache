import torch
import torch.nn as nn
from typing import Optional, Any, Dict, List
from .utils import VS, H, NL, HD, THETA, NH, NE, TOPK, EI, MASK_ID
from .rmsnorm import RMSNorm
from .layer import Layer
from .embedding import TokenEmbedding
from .rope import build_rope_freqs

class LLaDAMoESmall(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed_tokens = TokenEmbedding()
        self.layers = nn.ModuleList([Layer() for _ in range(NL)])
        self.norm = RMSNorm(H)
        self.lm_head = nn.Linear(H, VS, bias=False)

    def forward(
        self, 
        input_ids: torch.Tensor,
        cache_manager: Optional[Any] = None,
        caches: Optional[Dict[str, Any]] = None,
        k_step: Optional[int] = None,
        prompt_len: int = 0,
        update_ratio: float = 0.25
    ) -> torch.Tensor:
        B, T = input_ids.shape
        embed_cache = caches.get('embed') if caches is not None else None
        if embed_cache is not None and cache_manager is not None and k_step is not None:
            is_initial = cache_manager.is_initial_step(k_step)
            is_prompt_up = cache_manager.is_prompt_update(k_step)
            
            if is_initial or is_prompt_up:
                x = self.embed_tokens(input_ids)
                embed_cache.update_prompt(x[:, :prompt_len])
            else:
                cached_prompt_embeds = embed_cache.get_prompt()
                response_embeds = self.embed_tokens(input_ids[:, prompt_len:])
                x = torch.cat([cached_prompt_embeds, response_embeds], dim=1)
        else:
            x = self.embed_tokens(input_ids)
            
        
        cos, sin = build_rope_freqs(T, HD, THETA, input_ids.device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        
        layer_caches = caches.get('layers') if caches is not None else None
        for i, layer in enumerate(self.layers):
            layer_cache = layer_caches[i] if layer_caches is not None else None
            x = layer(
                x, 
                cos, 
                sin,
                layer_cache=layer_cache,
                cache_manager=cache_manager,
                k_step=k_step,
                prompt_len=prompt_len,
                update_ratio=update_ratio
            )
            
        return self.lm_head(self.norm(x))




####################################
if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    model = LLaDAMoESmall().to(torch.bfloat16).to(device)
    total = sum((p.numel() for p in model.parameters()))
    print(f'Parameters: {total / 1000000.0:.1f}M')
    print(f'  H={H}, NH={NH}, HD={HD}, NL={NL}, NE={NE}, TOPK={TOPK}, EI={EI}')
    ids = torch.full((1, 32), MASK_ID, dtype=torch.long, device=device)
    ids[0, :16] = torch.randint(0, 1000, (16,))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, 32, VS), f'Unexpected shape: {logits.shape}'
    print(f'Forward pass OK — logits shape: {logits.shape}')
    print(f'  Top-1 predicted token at position 16: {logits[0, 16].argmax().item()}')
    print('LLaDA-MoE-Small: all checks passed.')