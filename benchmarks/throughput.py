""" Measures throughput: how much work gets done per second, rather than
how long any single call takes. Built on top of latency.py, since
throughput is just work divided by time.
"""

import time
from dataclasses import dataclass

from .latency import measure_generation_latency, measure_diffusion_step_latency


@dataclass
class ThroughputResult:
    name: str
    value: float
    unit: str

    def __repr__(self):
        return f"{self.name}: {self.value:.2f} {self.unit}"


def measure_tokens_per_second(
    engine, prompt, gen_length: int = 64, warmup: int = 1, repeats: int = 5
) -> ThroughputResult:
    latency = measure_generation_latency(
        engine, prompt, gen_length=gen_length, warmup=warmup, repeats=repeats
    )
    seconds_per_call = latency.mean_ms / 1000.0
    tokens_per_sec = gen_length / seconds_per_call
    return ThroughputResult(name="tokens_per_second", value=tokens_per_sec, unit="tok/s")


def measure_steps_per_second(
    generator, tokens: list, step: int, editable: list, warmup: int = 3, repeats: int = 10
) -> ThroughputResult:
    latency = measure_diffusion_step_latency(
        generator, tokens, step, editable, warmup=warmup, repeats=repeats
    )
    seconds_per_step = latency.mean_ms / 1000.0
    steps_per_sec = 1.0 / seconds_per_step
    return ThroughputResult(name="diffusion_steps_per_second", value=steps_per_sec, unit="steps/s")


def measure_sequences_per_second(
    engine, prompts: list, gen_length: int = 64, warmup: int = 1
) -> ThroughputResult:
    for p in prompts[:warmup]:
        engine.generate(p, gen_length=gen_length)

    start = time.perf_counter()
    for p in prompts:
        engine.generate(p, gen_length=gen_length)
    end = time.perf_counter()

    total_seconds = end - start
    seqs_per_sec = len(prompts) / total_seconds
    return ThroughputResult(name="sequences_per_second", value=seqs_per_sec, unit="seq/s")