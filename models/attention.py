import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any

from .utils import H, NH, KVH, HD
from .rmsnorm import RMSNorm
from .rope import apply_rope, rotate_half
from cache.token_tracker import TokenTracker
from cache.selective_compute import gather_tokens, scatter_tokens

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, NH * HD, bias=False)
        self.k_proj = nn.Linear(H, KVH * HD, bias=False)
        self.v_proj = nn.Linear(H, KVH * HD, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.q_norm = RMSNorm(HD)
        self.k_norm = RMSNorm(HD)


    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        cache: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        k_step: Optional[int] = None,
        prompt_len: int = 0,
        update_ratio: float = 0.25
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, _ = x.shape

        if cache is None or cache_manager is None or k_step is None:
            q = self.q_proj(x).view(B, T, NH, HD)
            k = self.k_proj(x).view(B, T, KVH, HD)
            v = self.v_proj(x).view(B, T, KVH, HD)
            q = self.q_norm(q.reshape(-1, HD)).reshape(B, T, NH, HD)
            k = self.k_norm(k.reshape(-1, HD)).reshape(B, T, KVH, HD)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = apply_rope(q, k, cos, sin)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
            return self.o_proj(out.transpose(1, 2).reshape(B, T, H)), None

        is_initial = cache_manager.is_initial_step(k_step)
        is_prompt_up = cache_manager.is_prompt_update(k_step)
        is_resp_full = cache_manager.is_response_full_update(k_step)

        if is_initial:
            q = self.q_proj(x).view(B, T, NH, HD)
            k = self.k_proj(x).view(B, T, KVH, HD)
            v = self.v_proj(x).view(B, T, KVH, HD)
            q = self.q_norm(q.reshape(-1, HD)).reshape(B, T, NH, HD)
            k = self.k_norm(k.reshape(-1, HD)).reshape(B, T, KVH, HD)
            q = q.transpose(1, 2)  
            k = k.transpose(1, 2) 
            v = v.transpose(1, 2) 
            q, k = apply_rope(q, k, cos, sin)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
            attn_out = self.o_proj(out.transpose(1, 2).reshape(B, T, H))
            k_seq = k.transpose(1, 2) 
            v_seq = v.transpose(1, 2)
            cache.update_prompt(
                k_seq[:, :prompt_len], v_seq[:, :prompt_len],
                attn_out[:, :prompt_len], prompt_len
            )
            cache.update_response(
                k_seq[:, prompt_len:], v_seq[:, prompt_len:],
                attn_out[:, prompt_len:]
            )
            return attn_out, None

        attn_out_full = torch.empty(B, T, H, device=x.device, dtype=x.dtype)
        update_indices = None

        if is_prompt_up:
            x_p = x[:, :prompt_len, :]
            q_p = self.q_proj(x_p).view(B, prompt_len, NH, HD)
            k_p = self.k_proj(x_p).view(B, prompt_len, KVH, HD)
            v_p = self.v_proj(x_p).view(B, prompt_len, KVH, HD)
            q_p = self.q_norm(q_p.reshape(-1, HD)).reshape(B, prompt_len, NH, HD)
            k_p = self.k_norm(k_p.reshape(-1, HD)).reshape(B, prompt_len, KVH, HD)
            q_p_t = q_p.transpose(1, 2)
            k_p_t = k_p.transpose(1, 2)
            v_p_t = v_p.transpose(1, 2)
            q_p_t, k_p_t = apply_rope(q_p_t, k_p_t, cos[:prompt_len], sin[:prompt_len])
            needs_prompt_attn = True
        else:
            k_p_seq, v_p_seq, attn_out_p = cache.get_prompt()
            k_p_t = k_p_seq.transpose(1, 2)
            v_p_t = v_p_seq.transpose(1, 2)
            attn_out_full[:, :prompt_len, :] = attn_out_p
            needs_prompt_attn = False

        response_x = x[:, prompt_len:, :]
        B_r, T_r = response_x.shape[:2]

        if is_resp_full:
            q_r = self.q_proj(response_x).view(B, T_r, NH, HD)
            k_r = self.k_proj(response_x).view(B, T_r, KVH, HD)
            v_r = self.v_proj(response_x).view(B, T_r, KVH, HD)
            q_r = self.q_norm(q_r.reshape(-1, HD)).reshape(B, T_r, NH, HD)
            k_r = self.k_norm(k_r.reshape(-1, HD)).reshape(B, T_r, KVH, HD)
            q_r_t = q_r.transpose(1, 2)
            k_r_t = k_r.transpose(1, 2)
            v_r_t = v_r.transpose(1, 2)
            q_r_t, k_r_t = apply_rope(q_r_t, k_r_t, cos[prompt_len:], sin[prompt_len:])
            needs_resp_attn = 'full'
        else:
            # Compute full V_r (needed for token tracker and full-V cache replacement).
            v_r_all_seq = self.v_proj(response_x).view(B, T_r, KVH, HD)  # [B, T_r, KVH, HD]
            v_r_t = v_r_all_seq.transpose(1, 2)                           # [B, KVH, T_r, HD]

            tracker = TokenTracker(update_ratio=update_ratio)
            cached_v_response = cache.get_cached_v_response()
            indices_resp = tracker.verify(v_r_all_seq, cached_v_response)

            if indices_resp.shape[1] == 0:
                k_r_seq, v_r_seq, attn_out_r = cache.get_response()
                k_r_t = k_r_seq.transpose(1, 2)
                v_r_t = v_r_seq.transpose(1, 2)
                attn_out_full[:, prompt_len:, :] = attn_out_r
                needs_resp_attn = 'none'
            else:
                update_indices = indices_resp + prompt_len
                x_partial = gather_tokens(x, update_indices)

                q_partial = self.q_proj(x_partial).view(B, -1, NH, HD)
                q_partial = self.q_norm(q_partial.reshape(-1, HD)).reshape(B, -1, NH, HD)
                k_partial = self.k_proj(x_partial).view(B, -1, KVH, HD)
                k_partial = self.k_norm(k_partial.reshape(-1, HD)).reshape(B, -1, KVH, HD)

                q_partial_t = q_partial.transpose(1, 2)
                k_partial_t = k_partial.transpose(1, 2)

                cos_exp = cos.unsqueeze(0).expand(B, -1, -1)
                sin_exp = sin.unsqueeze(0).expand(B, -1, -1)
                cos_partial = gather_tokens(cos_exp, update_indices)
                sin_partial = gather_tokens(sin_exp, update_indices)
                cos_partial_uns = cos_partial.unsqueeze(1)
                sin_partial_uns = sin_partial.unsqueeze(1)

                q_partial_t = q_partial_t * cos_partial_uns + rotate_half(q_partial_t) * sin_partial_uns
                k_partial_t = k_partial_t * cos_partial_uns + rotate_half(k_partial_t) * sin_partial_uns

                k_partial_rope = k_partial_t.transpose(1, 2)

                k_r_cached_seq, v_r_cached_seq, _ = cache.get_response()
                k_r_seq = scatter_tokens(k_r_cached_seq, k_partial_rope, indices_resp)
                k_r_t = k_r_seq.transpose(1, 2)

                needs_resp_attn = 'partial'

        k_full_t = torch.cat([k_p_t, k_r_t], dim=2)
        v_full_t = torch.cat([v_p_t, v_r_t], dim=2)

        if needs_prompt_attn:
            out_p = F.scaled_dot_product_attention(q_p_t, k_full_t, v_full_t, attn_mask=None, is_causal=False)
            attn_out_p = self.o_proj(out_p.transpose(1, 2).reshape(B, prompt_len, H))
            attn_out_full[:, :prompt_len, :] = attn_out_p
            cache.update_prompt(k_p_t.transpose(1, 2), v_p_t.transpose(1, 2), attn_out_p, prompt_len)

        if needs_resp_attn == 'full':
            out_r = F.scaled_dot_product_attention(q_r_t, k_full_t, v_full_t, attn_mask=None, is_causal=False)
            attn_out_r = self.o_proj(out_r.transpose(1, 2).reshape(B, T_r, H))
            attn_out_full[:, prompt_len:, :] = attn_out_r
            cache.update_response(k_r_t.transpose(1, 2), v_r_t.transpose(1, 2), attn_out_r)

        elif needs_resp_attn == 'partial':
            out_r_partial = F.scaled_dot_product_attention(q_partial_t, k_full_t, v_full_t, attn_mask=None, is_causal=False)
            attn_out_r_partial = self.o_proj(out_r_partial.transpose(1, 2).reshape(B, -1, H))

            # k_r_seq is already the fully-scattered K_r — pass directly to avoid second scatter.
            cache.update_response_partial(k_r_seq, v_r_all_seq, attn_out_r_partial, indices_resp)

            _, _, attn_out_r = cache.get_response()
            attn_out_full[:, prompt_len:, :] = attn_out_r

        return attn_out_full, update_indices