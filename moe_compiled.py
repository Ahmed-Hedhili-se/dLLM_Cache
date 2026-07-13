"""
Step 2 — torch.compile the naive MoE layer and compare against eager.

The interesting part isn't the speedup number — it's WHERE compile helps and
where it can't. The router (linear -> softmax -> topk) is static-shape and
fuses beautifully. The per-expert gather/scatter loop has data-dependent
shapes (how many tokens land on expert e changes every forward pass), which
is exactly the kind of dynamic control flow that forces graph breaks.

Run with TORCH_LOGS to see graph breaks:
    TORCH_LOGS="graph_breaks" python moe_compiled.py

Run normally to just get the eager vs compiled numbers:
    python moe_compiled.py
"""

import torch
from moe_naive import NaiveMoE, _bench


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("NOTE: torch.compile's real payoff (kernel fusion, CUDA graphs) only")
        print("shows up on GPU. Run this on the H200 for numbers that matter.\n")

    torch.manual_seed(0)
    d_model, d_hidden, n_experts, top_k = 1024, 4096, 8, 2
    n_tokens = 2048

    model = NaiveMoE(d_model, d_hidden, n_experts, top_k).to(device).eval()
    x = torch.randn(n_tokens, d_model, device=device)

    with torch.no_grad():
        eager_out = model(x)
        eager_time = _bench(lambda t: model(t), x)
    print(f"Eager:    {eager_time * 1000:.3f} ms  ({n_tokens / eager_time:,.0f} tok/s)")

    compiled_model = torch.compile(model)
    with torch.no_grad():
        compiled_out = compiled_model(x)  # triggers compilation (excluded from timing)
        compiled_time = _bench(lambda t: compiled_model(t), x)
    print(f"Compiled: {compiled_time * 1000:.3f} ms  ({n_tokens / compiled_time:,.0f} tok/s)")

    max_diff = (eager_out - compiled_out).abs().max().item()
    print(f"\nSpeedup: {eager_time / compiled_time:.2f}x")
    print(f"Max abs diff eager vs compiled (correctness check): {max_diff:.2e}")

    print("\nTo see WHY it doesn't fully fuse, rerun with:")
    print('  TORCH_LOGS="graph_breaks" python moe_compiled.py')
    print("Look for breaks around the per-expert loop / nonzero() / index_add_ —")
    print("those are the data-dependent-shape ops.")


if __name__ == "__main__":
    main()
