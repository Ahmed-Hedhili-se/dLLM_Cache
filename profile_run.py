"""
Steps 3 & 4 — the script nsys/ncu will actually profile.

This file itself does nothing special — it just runs N forward passes of
each model variant so a profiler has something meaty to attach to. Run it
THROUGH nsys / ncu, not directly (though running it directly also works as
a smoke test).

--- Step 3: Nsight Systems (timeline, kernel launch gaps) ---
Run on the H200:
    nsys profile -o profiles/naive_moe \
        python profile_run.py --variant naive --iters 30

    nsys profile -o profiles/compiled_moe \
        python profile_run.py --variant compiled --iters 30

Open the resulting .nsys-rep in the Nsight Systems GUI (or `nsys stats
profiles/naive_moe.nsys-rep` for a text summary). What to look for:
    - naive: many short kernel launches per forward pass (one cluster per
      expert), with visible GPU-idle gaps between them (CPU-bound launch
      overhead dominating).
    - compiled: fewer, larger kernels for the parts that fused; the
      gather/scatter loop will likely still show as separate small kernels
      per expert (this is the graph break from Step 2, made visible).

--- Step 4: Nsight Compute (per-kernel occupancy, memory vs compute bound) ---
Pick ONE expert FFN kernel to drill into (e.g. the fc1 GEMM of expert 0).
Nsight Compute is much slower per-kernel, so profile a short run:
    ncu --set full -o profiles/expert_kernel \
        python profile_run.py --variant naive --iters 1

Open profiles/expert_kernel.ncu-rep. Key numbers to record:
    - "Compute (SM) Throughput %" vs "Memory Throughput %" — whichever is
      higher tells you if this kernel is compute-bound or memory-bound.
    - "Achieved Occupancy" — low occupancy on a small-batch expert GEMM is
      expected; that's the tensor-economics story (Step 5) showing up as a
      real measured number instead of a back-of-envelope estimate.
"""

import argparse
import torch

from moe_naive import NaiveMoE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["naive", "compiled"], default="naive")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--n_tokens", type=int, default=2048)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_hidden", type=int, default=4096)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found. nsys/ncu need an actual GPU (the H200) "
              "to produce meaningful profiles — this run will just smoke-test the code path.")

    torch.manual_seed(0)
    model = NaiveMoE(args.d_model, args.d_hidden, args.n_experts, args.top_k).to(device).eval()
    if args.variant == "compiled":
        model = torch.compile(model)

    x = torch.randn(args.n_tokens, args.d_model, device=device)

    # warmup (also triggers compilation if applicable) -- exclude from profiled region
    with torch.no_grad():
        for _ in range(3):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.iters):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    print(f"Done: variant={args.variant}, iters={args.iters}, device={device}")


if __name__ == "__main__":
    main()
