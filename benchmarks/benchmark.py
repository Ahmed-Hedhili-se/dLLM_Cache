""" Latency -> Throughput -> Correctness -> Profiler"""

import os
import sys


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


VOCAB_SIZE = 128
SEQ_LEN = 32
MASK_TOKEN_ID = 0
GEN_LENGTH = 16


def build_dummy_model():
    """
    A tiny random-logit 'model' so the benchmark suite is runnable
    immediately, with no real model wired up yet. Replace this with
    your actual model + tokenizer (e.g. LLaDA-MoE) when ready.
    """
    import torch

    def model_fn(tokens: list):
        seq_len = len(tokens)
        return torch.randn(seq_len, VOCAB_SIZE)

    return model_fn


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
    model_fn = build_dummy_model()

    scheduler = DiffusionScheduler(total_steps=10, schedule="cosine")
    generator = DiffusionGenerator(
        model_fn=model_fn,
        scheduler=scheduler,
        mask_token_id=MASK_TOKEN_ID,
        sampling_strategy="greedy",
    )
    engine = DiffusionInference(
        model=None,
        tokenizer=None,
        total_steps=10,
        schedule="cosine",
        mask_token_id=MASK_TOKEN_ID,
        sampling_strategy="greedy",
    )
    engine.generator.model_fn = model_fn

    prompt_ids = list(range(1, 9))  
    tokens = prompt_ids + [MASK_TOKEN_ID] * (SEQ_LEN - len(prompt_ids))
    editable = list(range(len(prompt_ids), SEQ_LEN))

    run_latency_benchmarks(model_fn, generator, tokens, editable)
    run_throughput_benchmarks(engine, generator, tokens, editable, prompt_ids)
    run_correctness_benchmarks(model_fn, tokens)
    run_profiler_benchmarks(engine, prompt_ids)


if __name__ == "__main__":
    main()