"""
Step 1 — Naive MoE baseline.

A standalone, minimal MoE FFN block: router (linear -> softmax -> top-k)
+ N independent expert FFNs + a naive python-loop dispatch (gather tokens
per expert, run FFN, scatter results back).

This is deliberately slow: one kernel-launch storm per expert, per forward
pass. The goal is not to be fast — it's to have something honest to profile
in Steps 3/4, and something to compare against once you fix the dispatch
(torch.compile in Step 2, grouped GEMM in Step 6).

Run directly to sanity-check correctness + get a naive wall-clock baseline:
    python moe_naive.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single expert: a plain 2-layer FFN, same shape as a transformer MLP block."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class NaiveMoE(nn.Module):
    """
    Top-k MoE layer with a naive per-expert dispatch loop.

    Args:
        d_model:   hidden size of the token representations
        d_hidden:  hidden size inside each expert FFN
        n_experts: total number of experts
        top_k:     experts activated per token (e.g. 2, like LLaDA-MoE-ish top-k routing)
    """

    def __init__(self, d_model: int = 1024, d_hidden: int = 4096, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_tokens, d_model) — flattened tokens (batch*seq already merged upstream)
        returns: (n_tokens, d_model)
        """
        n_tokens, d_model = x.shape

        # --- Routing ---
        router_logits = self.router(x)                       # (n_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)       # (n_tokens, n_experts)
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (n_tokens, top_k)
        # renormalize so the top_k weights sum to 1 per token (standard MoE practice)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)

        # --- Naive dispatch: one python-level loop per expert ---
        # For each expert, find every (token, slot) pair routed to it, gather those
        # tokens, run the FFN as ONE small batched matmul, scatter-add back weighted
        # by the router probability. This is exactly the pattern that produces many
        # small, serialized kernel launches — the thing nsys will show you in Step 3.
        for e in range(self.n_experts):
            # mask: which (token, slot) pairs picked this expert
            match = (topk_idx == e)                # (n_tokens, top_k) bool
            if not match.any():
                continue
            token_ids, slot_ids = match.nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue

            expert_input = x[token_ids]             # (n_selected, d_model) — gather
            expert_out = self.experts[e](expert_input)  # small batched GEMM

            weights = topk_probs[token_ids, slot_ids].unsqueeze(-1)  # (n_selected, 1)
            output.index_add_(0, token_ids, expert_out * weights)    # scatter-add

        return output


def _bench(fn, *args, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(0)
    d_model, d_hidden, n_experts, top_k = 1024, 4096, 8, 2
    n_tokens = 2048  # e.g. a few concurrent requests worth of tokens

    model = NaiveMoE(d_model, d_hidden, n_experts, top_k).to(device).eval()
    x = torch.randn(n_tokens, d_model, device=device)

    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}  (sanity check: {out.shape == x.shape})")

    with torch.no_grad():
        avg_s = _bench(lambda t: model(t), x)
    print(f"Naive MoE avg forward: {avg_s * 1000:.3f} ms  "
          f"({n_tokens / avg_s:,.0f} tokens/sec)")
