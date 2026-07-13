"""
Step 5 — Tensor economics: by-hand roofline analysis of one expert FFN.

Closes the open loop from the "Tensor Economics: MoE Inference" article —
instead of reading the argument, compute it yourself for this exact layer
and see where it lands relative to the H200's roofline ridge point.

Core idea (arithmetic intensity / roofline model):
    - Every GPU kernel is either compute-bound or memory-bound.
    - Arithmetic Intensity (AI) = FLOPs / Bytes moved.
    - Ridge point = peak_FLOPs / peak_bandwidth. If a kernel's AI is below
      the ridge point, it's memory-bound (the GPU is waiting on data, not
      computing) — no matter how fast the compute units are.
    - MoE inference at low batch size is the textbook memory-bound case:
      each expert only sees a handful of tokens per forward pass, so the
      cost of MOVING the expert's weights dwarfs the cost of the matmul
      itself.

Run:
    python tensor_economics.py
"""

# ---- H200 specs (public spec sheet numbers) ----
H200_BF16_TFLOPS = 989.0          # dense BF16 tensor-core FLOPs (no sparsity), TFLOP/s
H200_HBM_BANDWIDTH_GBPS = 4800.0  # HBM3e bandwidth, GB/s
BYTES_PER_PARAM_BF16 = 2

RIDGE_POINT = (H200_BF16_TFLOPS * 1e12) / (H200_HBM_BANDWIDTH_GBPS * 1e9)  # FLOPs/byte


def expert_ffn_analysis(d_model: int, d_hidden: int, n_tokens: int):
    """
    One expert FFN: fc1 (d_model -> d_hidden) + gelu + fc2 (d_hidden -> d_model).
    Ignoring bias terms (negligible).
    """
    # --- FLOPs ---
    # A (n_tokens x d_model) @ (d_model x d_hidden) matmul costs 2 * n_tokens * d_model * d_hidden FLOPs
    # (the factor of 2 is multiply + add per MAC).
    flops_fc1 = 2 * n_tokens * d_model * d_hidden
    flops_fc2 = 2 * n_tokens * d_hidden * d_model
    total_flops = flops_fc1 + flops_fc2

    # --- Bytes moved (weight-dominated at low batch: reading the weights once
    #     costs far more than reading/writing the small activation tensors) ---
    weight_params = (d_model * d_hidden) + (d_hidden * d_model)  # fc1 + fc2 weights
    weight_bytes = weight_params * BYTES_PER_PARAM_BF16

    activation_bytes = 2 * (
        n_tokens * d_model +        # read input
        n_tokens * d_hidden +       # write/read hidden activation
        n_tokens * d_model          # write output
    ) * BYTES_PER_PARAM_BF16

    total_bytes = weight_bytes + activation_bytes

    arithmetic_intensity = total_flops / total_bytes
    is_memory_bound = arithmetic_intensity < RIDGE_POINT

    # Roofline-predicted time: whichever bound (compute or memory) is the bottleneck
    time_compute_bound_s = total_flops / (H200_BF16_TFLOPS * 1e12)
    time_memory_bound_s = total_bytes / (H200_HBM_BANDWIDTH_GBPS * 1e9)
    predicted_time_s = max(time_compute_bound_s, time_memory_bound_s)

    return {
        "n_tokens": n_tokens,
        "total_flops": total_flops,
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "total_bytes": total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "is_memory_bound": is_memory_bound,
        "time_compute_bound_us": time_compute_bound_s * 1e6,
        "time_memory_bound_us": time_memory_bound_s * 1e6,
        "predicted_time_us": predicted_time_s * 1e6,
        "predicted_throughput_tok_s": n_tokens / predicted_time_s,
    }


def main():
    d_model, d_hidden = 1024, 4096
    print(f"H200 ridge point: {RIDGE_POINT:.1f} FLOPs/byte")
    print(f"  (below this AI -> memory-bound, above -> compute-bound)\n")

    # Sweep tokens-per-expert to show the memory-bound -> compute-bound transition.
    # At top_k=2 over 8 experts, avg tokens/expert ~= n_tokens * top_k / n_experts.
    print(f"{'tokens/expert':>14} | {'AI (FLOP/B)':>12} | {'bound':>10} | {'pred time (us)':>15} | {'pred tok/s':>12}")
    print("-" * 75)
    for n_tokens in [1, 4, 16, 64, 256, 1024, 4096, 16384]:
        r = expert_ffn_analysis(d_model, d_hidden, n_tokens)
        bound = "memory" if r["is_memory_bound"] else "compute"
        print(f"{n_tokens:>14} | {r['arithmetic_intensity']:>12.2f} | {bound:>10} | "
              f"{r['predicted_time_us']:>15.2f} | {r['predicted_throughput_tok_s']:>12,.0f}")

    print("\nTakeaway: at low tokens/expert (the realistic case at low concurrency,")
    print("since each expert only gets a fraction of a small batch via top-k routing),")
    print("the kernel is deep in memory-bound territory — you're paying to move the")
    print("expert's weights from HBM regardless of how few tokens use them. This is")
    print("exactly the argument for grouped GEMM / batching tokens across experts:")
    print("it doesn't reduce FLOPs, it amortizes the weight-load cost over more tokens,")
    print("pushing arithmetic intensity up and to the right on the roofline.")


if __name__ == "__main__":
    main()
