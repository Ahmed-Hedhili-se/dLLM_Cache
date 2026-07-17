""" Latency -> Throughput -> Correctness -> Profiler"""

import os
import sys
import types

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import configs.cache_config as cache_config
import configs.inference_config as inference_config
from models.model import LLaDAMoESmall
from models.utils import NL, MASK_ID, H, NE, TOPK
from cache import CacheManager, EmbeddingCache, AttentionCache, MoECache


if __package__ in (None, ""):
    from benchmarks.latency import (
        measure_forward_latency,
        measure_diffusion_step_latency,
        measure_generation_latency,
    )
    from benchmarks.throughput import (
        measure_tokens_per_second,
        measure_steps_per_second,
        measure_sequences_per_second,
    )
    from benchmarks.correctness import compare_logits, compare_token_sequences
    from benchmarks.profiler import DiffusionProfiler
else:
    from .latency import (
        measure_forward_latency,
        measure_diffusion_step_latency,
        measure_generation_latency,
    )
    from .throughput import (
        measure_tokens_per_second,
        measure_steps_per_second,
        measure_sequences_per_second,
    )
    from .correctness import compare_logits, compare_token_sequences
    from .profiler import DiffusionProfiler

from diffusion import DiffusionScheduler, DiffusionGenerator, DiffusionInference


# ── Fixed: SEQ_LEN must be > prompt_len (128) so editable region is non-empty.
# Old value (32) caused editable=[] and _denoise_step short-circuited at step>0.
SEQ_LEN = 256
GEN_LENGTH = 16
CORRECTNESS_GEN_LENGTH = 64  # Larger gen_length for correctness sweeps



def print_benchmark_params(prompt_len: int):
    """Print all benchmark parameters explicitly so results are reproducible."""
    print("=" * 60)
    print("  dLLM-Cache Benchmark — Parameter Summary")
    print("=" * 60)
    print(f"  Model Architecture:")
    print(f"    H (hidden dim)     = {H}")
    print(f"    NL (num layers)    = {NL}")
    print(f"    NE (num experts)   = {NE}")
    print(f"    TOPK (active exp.) = {TOPK}")
    print(f"  Cache Parameters:")
    print(f"    K_p (prompt refresh interval)   = {cache_config.KP}")
    print(f"    K_r (response refresh interval) = {cache_config.KR}")
    print(f"    ρ (UPDATE_RATIO)                = {cache_config.UPDATE_RATIO}")
    print(f"    NUM_DIFFUSION_STEPS             = {cache_config.NUM_DIFFUSION_STEPS}")
    print(f"  Sequence Config:")
    print(f"    SEQ_LEN            = {SEQ_LEN}")
    print(f"    prompt_len         = {prompt_len}")
    print(f"    num_editable       = {SEQ_LEN - prompt_len}")
    print(f"    GEN_LENGTH (lat.)  = {GEN_LENGTH}")
    print(f"    GEN_LENGTH (corr.) = {CORRECTNESS_GEN_LENGTH}")
    print(f"  Inference Config:")
    print(f"    batch_size         = {inference_config.BATCH_SIZE}")
    print(f"    device             = {inference_config.DEVICE}")
    print(f"    dtype              = {'bfloat16' if inference_config.USE_BFLOAT16 else 'float32'}")
    print("=" * 60)


def run_latency_benchmarks(baseline_model_fn, cached_model_fn, baseline_generator, cached_generator, tokens, editable, reset_caches_fn):
    print("\n=== Latency ===")
    
    # 1. Forward Latency
    print("Measuring single forward pass latency...")
    base_fwd = measure_forward_latency(baseline_model_fn, tokens)
    reset_caches_fn()
    cache_fwd = measure_forward_latency(cached_model_fn, tokens)
    print(f"  Baseline Forward: {base_fwd.mean_ms:.3f}ms")
    print(f"  Cached Forward (Initial/Full): {cache_fwd.mean_ms:.3f}ms")
    print(f"  Speedup: {base_fwd.mean_ms / cache_fwd.mean_ms:.2f}x")
    print()
    
    # 2. Diffusion Step Latency (Step 0 - Full Refresh)
    print("Measuring diffusion step 0 (Full Refresh) latency...")
    base_step0 = measure_diffusion_step_latency(baseline_generator, tokens, step=0, editable=editable)
    reset_caches_fn()
    cache_step0 = measure_diffusion_step_latency(cached_generator, tokens, step=0, editable=editable)
    print(f"  Baseline Step 0: {base_step0.mean_ms:.3f}ms")
    print(f"  Cached Step 0:   {cache_step0.mean_ms:.3f}ms")
    print(f"  Speedup: {base_step0.mean_ms / cache_step0.mean_ms:.2f}x")
    print()
    
    # 3. Diffusion Step Latency (Step 64 - Adaptive Partial Update)
    print("Measuring diffusion step 64 (Adaptive Partial Update) latency...")
    base_step64 = measure_diffusion_step_latency(baseline_generator, tokens, step=64, editable=editable)
    
    # Initialize cache by running step 0 first
    reset_caches_fn()
    cached_generator._denoise_step(tokens, 0, editable)
    
    cache_step64 = measure_diffusion_step_latency(cached_generator, tokens, step=64, editable=editable)
    print(f"  Baseline Step 64: {base_step64.mean_ms:.3f}ms")
    print(f"  Cached Step 64:   {cache_step64.mean_ms:.3f}ms")
    print(f"  Speedup: {base_step64.mean_ms / cache_step64.mean_ms:.2f}x")


def run_throughput_benchmarks(baseline_engine, cached_engine, tokens, editable, prompt, reset_caches_fn):
    print("\n=== Throughput ===")
    
    # 1. Diffusion Steps Per Second
    print("Measuring steps per second...")
    base_steps = measure_steps_per_second(baseline_engine.generator, tokens, step=0, editable=editable)
    reset_caches_fn()
    cache_steps = measure_steps_per_second(cached_engine.generator, tokens, step=0, editable=editable)
    print(f"  Baseline: {base_steps.value:.2f} steps/s")
    print(f"  Cached:   {cache_steps.value:.2f} steps/s")
    print(f"  Speedup:  {cache_steps.value / base_steps.value:.2f}x")
    print()
    
    # 2. Tokens Per Second (End-to-End Generation)
    print("Measuring tokens per second (end-to-end)...")
    base_tokens = measure_tokens_per_second(baseline_engine, prompt, gen_length=GEN_LENGTH)
    reset_caches_fn()
    cache_tokens = measure_tokens_per_second(cached_engine, prompt, gen_length=GEN_LENGTH)
    print(f"  Baseline: {base_tokens.value:.2f} tok/s")
    print(f"  Cached:   {cache_tokens.value:.2f} tok/s")
    print(f"  Speedup:  {cache_tokens.value / base_tokens.value:.2f}x")


def _analyze_mismatch_positions(
    baseline_logits: torch.Tensor,
    cached_logits: torch.Tensor,
    baseline_tokens: list,
    cached_tokens: list,
    prompt_len: int,
):
    """
    For each mismatched generated token, report position, token IDs,
    and whether it was a near-tie or a confident divergence.
    """
    gen_start = prompt_len
    mismatches = []
    for i in range(gen_start, min(len(baseline_tokens), len(cached_tokens))):
        if baseline_tokens[i] != cached_tokens[i]:
            info = {"position": i, "baseline_token": baseline_tokens[i], "cached_token": cached_tokens[i]}
            # If logits are available, compute probability gap
            if baseline_logits is not None and cached_logits is not None:
                bl = baseline_logits[i].float()
                cl = cached_logits[i].float()
                bp = F.softmax(bl, dim=-1)
                cp = F.softmax(cl, dim=-1)
                info["baseline_top1_prob"] = bp.max().item()
                info["cached_top1_prob"] = cp.max().item()
                info["baseline_argmax"] = bl.argmax().item()
                info["cached_argmax"] = cl.argmax().item()
                info["prob_gap"] = abs(bp.max().item() - cp.max().item())
                info["near_tie"] = info["prob_gap"] < 0.1
            mismatches.append(info)
    return mismatches


def run_correctness_benchmarks(
    baseline_model_fn, cached_model_fn,
    baseline_engine, cached_engine,
    tokens, prompt, prompt_len,
    reset_caches_fn, device,
):
    print("\n=== Correctness ===")
    
    # ── 1. Logits comparison at ρ=1.0 (sanity check — should always pass)
    print("Comparing logits (single forward at update_ratio=1.0)...")
    base_logits = baseline_model_fn(tokens)
    reset_caches_fn()
    
    original_ratio = cache_config.UPDATE_RATIO
    cache_config.UPDATE_RATIO = 1.0
    cached_logits = cached_model_fn(tokens)
    cache_config.UPDATE_RATIO = original_ratio
    
    res = compare_logits(base_logits, cached_logits, name="logits_equivalence_ratio_1.0")
    print(f"  {res}")
    
    # ── 2. Multi-ρ logits sweep
    ratios = [0.1, 0.25, 0.5, 0.9, 1.0]
    print(f"\n--- Multi-ρ Logits Sweep (ρ ∈ {ratios}) ---")
    for rho in ratios:
        reset_caches_fn()
        cache_config.UPDATE_RATIO = rho
        # Run step 0 first to populate caches
        cached_model_fn(tokens)
        reset_caches_fn()
        
        bl = baseline_model_fn(tokens)
        reset_caches_fn()
        cache_config.UPDATE_RATIO = rho
        cl = cached_model_fn(tokens)
        
        res_rho = compare_logits(bl, cl, name=f"logits_rho_{rho}")
        print(f"  ρ={rho:.2f}: {res_rho}")
    cache_config.UPDATE_RATIO = original_ratio

    # ── 3. Multi-ρ token generation sweep with mismatch analysis
    print(f"\n--- Multi-ρ Token Generation Sweep (gen_length={CORRECTNESS_GEN_LENGTH}) ---")
    mismatch_counts = {}
    for rho in ratios:
        cache_config.UPDATE_RATIO = rho
        
        reset_caches_fn()
        base_gen = baseline_engine.generate(prompt, gen_length=CORRECTNESS_GEN_LENGTH)
        
        reset_caches_fn()
        cached_gen = cached_engine.generate(prompt, gen_length=CORRECTNESS_GEN_LENGTH)
        
        res_tok = compare_token_sequences(base_gen, cached_gen, name=f"tokens_rho_{rho}")
        n_mismatch = sum(1 for a, b in zip(base_gen, cached_gen) if a != b)
        mismatch_counts[rho] = n_mismatch
        pct = (n_mismatch / CORRECTNESS_GEN_LENGTH * 100) if CORRECTNESS_GEN_LENGTH > 0 else 0
        print(f"  ρ={rho:.2f}: {n_mismatch}/{CORRECTNESS_GEN_LENGTH} mismatches ({pct:.1f}%)  {res_tok}")
        
        # Detailed mismatch position analysis
        if n_mismatch > 0:
            # Get logits for analysis
            reset_caches_fn()
            full_base = prompt + [MASK_ID] * CORRECTNESS_GEN_LENGTH
            bl_logits = baseline_model_fn(full_base)
            reset_caches_fn()
            cache_config.UPDATE_RATIO = rho
            cl_logits = cached_model_fn(full_base)
            
            mismatches = _analyze_mismatch_positions(
                bl_logits, cl_logits, base_gen, cached_gen, len(prompt)
            )
            for m in mismatches:
                tie_str = "NEAR-TIE" if m.get("near_tie", False) else "CONFIDENT-DIVERGENCE"
                print(f"    pos={m['position']}: baseline_tok={m['baseline_token']} "
                      f"cached_tok={m['cached_token']}  "
                      f"baseline_p1={m.get('baseline_top1_prob', '?'):.4f}  "
                      f"cached_p1={m.get('cached_top1_prob', '?'):.4f}  "
                      f"gap={m.get('prob_gap', '?'):.4f}  [{tie_str}]")

    cache_config.UPDATE_RATIO = original_ratio

    # ── 4. Monotonicity check
    print(f"\n--- Monotonicity Check ---")
    print(f"  Mismatch counts by ρ: {mismatch_counts}")
    sorted_rhos = sorted(mismatch_counts.keys())
    monotonic = all(
        mismatch_counts[sorted_rhos[i]] >= mismatch_counts[sorted_rhos[i + 1]]
        for i in range(len(sorted_rhos) - 1)
    )
    if monotonic:
        print("  ✓ Mismatches decrease monotonically as ρ → 1.0 (consistent with expected lossy approximation)")
    else:
        print("  ✗ WARNING: Mismatches do NOT decrease monotonically — possible cache/scatter bug")

    if mismatch_counts.get(0.9, 0) > 0:
        print(f"  ⚠ FLAG: {mismatch_counts[0.9]} mismatches at ρ=0.9 — almost all tokens should match baseline.")
        print(f"          This may indicate a bug in cache invalidation / scatter logic (stale K/V leaking).")
    if mismatch_counts.get(1.0, 0) > 0:
        print(f"  ✗ CRITICAL: {mismatch_counts[1.0]} mismatches at ρ=1.0 — full update should be exact!")
        print(f"              This is a definite bug, not approximation error.")
    else:
        print("  ✓ ρ=1.0 produces zero mismatches (full update is exact, as expected)")


def run_profiler_benchmarks(engine, prompt):
    print("\n=== Profiler ===")
    try:
        profiler = DiffusionProfiler()
    except ImportError:
        print("torch not available, skipping profiler")
        return

    with profiler.profile():
        engine.generate(prompt, gen_length=GEN_LENGTH)
    profiler.print_summary(row_limit=10)


def main():
    device = inference_config.DEVICE
    print(f"Initializing 7B Model (mapped to custom LLaDAMoE architecture) on {device}...")
    model = LLaDAMoESmall()
    from models.loader import get_hf_weights_path, load_hf_weights_into_custom_model
    try:
        weights_path = get_hf_weights_path()
        model = load_hf_weights_into_custom_model(model, weights_path)
    except Exception as e:
        print(f"Failed to load weights automatically: {e}")
        print("Please ensure you have run download_weigts.py or have enough disk space.")
    if inference_config.USE_BFLOAT16:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model.eval()

    cache_manager = CacheManager(
        k_p=cache_config.KP,
        k_r=cache_config.KR,
        total_steps=cache_config.NUM_DIFFUSION_STEPS
    )

    caches = {'embed': EmbeddingCache() if cache_config.ENABLE_EMBEDDING_CACHE else None}
    layer_caches = []
    for _ in range(NL):
        layer_caches.append({
            'attn': AttentionCache() if cache_config.ENABLE_ATTENTION_CACHE else None,
            'mlp': MoECache() if cache_config.ENABLE_MOE_CACHE else None
        })
    caches['layers'] = layer_caches

    prompt_ids = list(range(1, 129))
    prompt_len = len(prompt_ids)

    # Print explicit benchmark parameters
    print_benchmark_params(prompt_len)

    # 1. Baseline Engine
    baseline_engine = DiffusionInference(
        model=model,
        tokenizer=None,
        total_steps=cache_config.NUM_DIFFUSION_STEPS,
        mask_token_id=MASK_ID,
        sampling_strategy="greedy",
    )
    def baseline_model_fn(tokens: list):
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
        return logits[0]
    
    baseline_engine.generator.model_fn = baseline_model_fn
    baseline_engine._model_fn = types.MethodType(lambda self, tokens: baseline_model_fn(tokens), baseline_engine)

    # 2. Cached Engine
    cached_engine = DiffusionInference(
        model=model,
        tokenizer=None,
        total_steps=cache_config.NUM_DIFFUSION_STEPS,
        mask_token_id=MASK_ID,
        sampling_strategy="greedy",
    )

    original_denoise = cached_engine.generator._denoise_step
    def hooked_denoise(self, tokens, step, editable):
        self._current_step = step
        return original_denoise(tokens, step, editable)
    cached_engine.generator._denoise_step = types.MethodType(hooked_denoise, cached_engine.generator)

    def cached_model_fn(tokens: list):
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        k_step = cache_config.NUM_DIFFUSION_STEPS - getattr(cached_engine.generator, "_current_step", 0)
        
        with torch.no_grad():
            output = model(
                input_ids,
                cache_manager=cache_manager,
                caches=caches,
                k_step=k_step,
                prompt_len=prompt_len,
                update_ratio=cache_config.UPDATE_RATIO
            )
            logits = output.logits if hasattr(output, "logits") else output
        return logits[0]

    cached_engine.generator.model_fn = cached_model_fn
    cached_engine._model_fn = types.MethodType(lambda self, tokens: cached_model_fn(tokens), cached_engine)

    def reset_caches_fn():
        if caches.get('embed') is not None:
            caches['embed'].reset()
        for layer_cache in caches.get('layers', []):
            if layer_cache.get('attn') is not None:
                layer_cache['attn'].reset()
            if layer_cache.get('mlp') is not None:
                layer_cache['mlp'].reset()

    tokens = prompt_ids + [MASK_ID] * (SEQ_LEN - len(prompt_ids))
    editable = list(range(len(prompt_ids), SEQ_LEN))

    # Sanity check: editable must be non-empty for step-latency benchmarks
    assert len(editable) > 0, (
        f"BUG: editable region is empty! SEQ_LEN={SEQ_LEN} must be > prompt_len={prompt_len}. "
        f"Got editable=range({len(prompt_ids)}, {SEQ_LEN})."
    )
    print(f"\n[Sanity] editable positions: {len(editable)} (range [{editable[0]}, {editable[-1]}])")

    run_latency_benchmarks(
        baseline_model_fn, cached_model_fn,
        baseline_engine.generator, cached_engine.generator,
        tokens, editable, reset_caches_fn
    )
    
    run_throughput_benchmarks(
        baseline_engine, cached_engine,
        tokens, editable, prompt_ids, reset_caches_fn
    )
    
    run_correctness_benchmarks(
        baseline_model_fn, cached_model_fn,
        baseline_engine, cached_engine,
        tokens, prompt_ids, prompt_len,
        reset_caches_fn, device
    )
    
    run_profiler_benchmarks(cached_engine, prompt_ids)


if __name__ == "__main__":
    main()