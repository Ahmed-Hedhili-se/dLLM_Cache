""" Latency -> Throughput -> Correctness -> Profiler"""

import os
import sys
import types

import torch
import configs.cache_config as cache_config
import configs.inference_config as inference_config
from models.model import LLaDAMoESmall
from models.utils import NL, MASK_ID
from cache import CacheManager, EmbeddingCache, AttentionCache, MoECache


if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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



def run_latency_benchmarks(model_fn, generator, tokens, editable):
    print("\n=== Latency ===")
    print(measure_forward_latency(model_fn, tokens))
    print(measure_diffusion_step_latency(generator, tokens, step=0, editable=editable))


def run_throughput_benchmarks(engine, generator, tokens, editable, prompt):
    print("\n=== Throughput ===")
    print(measure_steps_per_second(generator, tokens, step=0, editable=editable))
    print(measure_tokens_per_second(engine, prompt, gen_length=GEN_LENGTH))


def run_correctness_benchmarks(model_fn, tokens):
    print("\n=== Correctness ===")
    baseline_logits = model_fn(tokens)
    optimized_logits = baseline_logits.clone()
    print(compare_logits(baseline_logits, optimized_logits, name="baseline_vs_optimized"))
    print(compare_token_sequences([1, 2, 3, 4], [1, 2, 3, 4], name="token_sequence_demo"))


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

    engine = DiffusionInference(
        model=model,
        tokenizer=None,
        total_steps=cache_config.NUM_DIFFUSION_STEPS,
        mask_token_id=MASK_ID,
        sampling_strategy="greedy",
    )

    # Hook denoise_step to capture the current step
    original_denoise = engine.generator._denoise_step
    def hooked_denoise(self, tokens, step, editable):
        self._current_step = step
        return original_denoise(tokens, step, editable)
    engine.generator._denoise_step = types.MethodType(hooked_denoise, engine.generator)

    def custom_model_fn(tokens: list):
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        k_step = cache_config.NUM_DIFFUSION_STEPS - getattr(engine.generator, "_current_step", 0)
        
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

    engine.generator.model_fn = custom_model_fn
    engine._model_fn = types.MethodType(lambda self, tokens: custom_model_fn(tokens), engine)

    tokens = prompt_ids + [MASK_ID] * (SEQ_LEN - len(prompt_ids))
    editable = list(range(len(prompt_ids), SEQ_LEN))

    run_latency_benchmarks(custom_model_fn, engine.generator, tokens, editable)
    run_throughput_benchmarks(engine, engine.generator, tokens, editable, prompt_ids)
    run_correctness_benchmarks(custom_model_fn, tokens)
    run_profiler_benchmarks(engine, prompt_ids)


if __name__ == "__main__":
    main()