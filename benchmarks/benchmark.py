""" Latency -> Throughput -> Correctness -> Profiler"""

import os
import sys
import types

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import configs.cache_config as cache_config
import configs.inference_config as inference_config
from models.model import LLaDAMoESmall
from models.utils import NL, MASK_ID
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


SEQ_LEN = 32
GEN_LENGTH = 16



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
    
    # 3. Diffusion Step Latency (Step 1 - Adaptive Partial Update)
    print("Measuring diffusion step 1 (Adaptive Partial Update) latency...")
    base_step1 = measure_diffusion_step_latency(baseline_generator, tokens, step=1, editable=editable)
    
    # Initialize cache by running step 0 first
    reset_caches_fn()
    cached_generator._denoise_step(tokens, 0, editable)
    
    cache_step1 = measure_diffusion_step_latency(cached_generator, tokens, step=1, editable=editable)
    print(f"  Baseline Step 1: {base_step1.mean_ms:.3f}ms")
    print(f"  Cached Step 1:   {cache_step1.mean_ms:.3f}ms")
    print(f"  Speedup: {base_step1.mean_ms / cache_step1.mean_ms:.2f}x")


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


def run_correctness_benchmarks(baseline_model_fn, cached_model_fn, baseline_engine, cached_engine, tokens, prompt, reset_caches_fn):
    print("\n=== Correctness ===")
    
    # 1. Logits comparison (at update_ratio=1.0)
    print("Comparing logits (single forward at update_ratio=1.0)...")
    base_logits = baseline_model_fn(tokens)
    reset_caches_fn()
    
    original_ratio = cache_config.UPDATE_RATIO
    cache_config.UPDATE_RATIO = 1.0
    cached_logits = cached_model_fn(tokens)
    cache_config.UPDATE_RATIO = original_ratio
    
    res = compare_logits(base_logits, cached_logits, name="logits_equivalence_ratio_1.0")
    print(f"  {res}")
    
    # 2. Token sequence comparison
    print("\nComparing generated token sequences (end-to-end)...")
    base_gen_tokens = baseline_engine.generate(prompt, gen_length=GEN_LENGTH)
    reset_caches_fn()
    cached_gen_tokens = cached_engine.generate(prompt, gen_length=GEN_LENGTH)
    
    res_tokens = compare_token_sequences(base_gen_tokens, cached_gen_tokens, name="generated_tokens_similarity")
    print(f"  {res_tokens}")
    print(f"  Baseline generated: {base_gen_tokens}")
    print(f"  Cached generated:   {cached_gen_tokens}")


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
    print(f"Initializing LLaDAMoESmall model on {device}...")
    model = LLaDAMoESmall()
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

    prompt_ids = list(range(1, 9))
    prompt_len = len(prompt_ids)

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
        tokens, prompt_ids, reset_caches_fn
    )
    
    run_profiler_benchmarks(cached_engine, prompt_ids)


if __name__ == "__main__":
    main()