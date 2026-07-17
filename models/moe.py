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

        if is_initial:
            out = self._compute_moe(x)
            out_p = out[:, :prompt_len]
            out_r = out[:, prompt_len:]
            cache.update_prompt(out_p, prompt_len)
            cache.update_response(out_r)
            return out

        # Both halves are always written before this tensor is returned — use empty.
        out_full = torch.empty_like(x)


        if is_prompt_up:
            out_p = self._compute_moe(x[:, :prompt_len])
            out_full[:, :prompt_len] = out_p
            cache.update_prompt(out_p, prompt_len)
        else:
            out_full[:, :prompt_len] = cache.get_prompt()

        if is_resp_full:
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
        """
        Sparse MoE dispatch with a single CPU-GPU sync.

        Original approach used ``F.one_hot + torch.where`` which forces one
        CPU-GPU synchronisation per expert (NE=64 syncs per layer per step).
        This version does ONE sync (``bincount().cpu()``) to bring token counts
        to CPU, then slices a pre-sorted contiguous buffer — zero additional
        syncs for the 64 expert forward passes.
        """
        B, T, _ = x.shape
        N = B * T
        x_flat = x.view(N, H)

        # ── Gate ──────────────────────────────────────────────────────────────
        gate_logits = self.gate(x_flat)                                   # [N, NE]
        routing_weights = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
        top_weights, top_experts = torch.topk(routing_weights, TOPK, dim=-1)  # [N, TOPK]
        top_weights = top_weights.to(x.dtype)

        # ── Build dispatch table ───────────────────────────────────────────────
        # Each of the N×TOPK dispatch entries maps to (token_index, expert_index, weight)
        token_2d   = torch.arange(N, device=x.device).unsqueeze(1).expand(N, TOPK)
        token_flat = token_2d.reshape(-1)        # [N*TOPK]
        expert_flat = top_experts.reshape(-1)     # [N*TOPK]
        weight_flat = top_weights.reshape(-1)     # [N*TOPK]

        # Sort by expert so each expert's tokens are contiguous in memory
        perm           = torch.argsort(expert_flat, stable=True)
        expert_sorted  = expert_flat[perm]        # [N*TOPK]
        token_sorted   = token_flat[perm]         # [N*TOPK]
        weight_sorted  = weight_flat[perm]        # [N*TOPK]

        # Gather token features in sorted (expert-contiguous) order
        x_dispatched = x_flat[token_sorted]       # [N*TOPK, H]

        # ── ONE CPU-GPU sync: bring per-expert token counts to CPU ─────────────
        counts = torch.bincount(expert_sorted, minlength=NE).cpu()  # single sync

        # ── Expert forward passes: contiguous slice per expert, no extra syncs ──
        out_dispatched = torch.empty_like(x_dispatched)
        offset = 0
        for e_idx in range(NE):
            n_e = counts[e_idx].item()    # CPU read — no GPU sync
            if n_e > 0:
                out_dispatched[offset : offset + n_e] = \
                    self.experts[e_idx](x_dispatched[offset : offset + n_e])
            offset += n_e

        # ── Weight and accumulate ──────────────────────────────────────────────
        out_dispatched.mul_(weight_sorted.unsqueeze(-1))
        out = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        out.index_add_(0, token_sorted, out_dispatched)

        return out.view(B, T, H)