"""benchmarks package :)"""

from .latency import (
    LatencyResult,
    measure_forward_latency,
    measure_diffusion_step_latency,
    measure_generation_latency,
)
from .throughput import (
    ThroughputResult,
    measure_tokens_per_second,
    measure_steps_per_second,
    measure_sequences_per_second,
)
from .correctness import (
    CorrectnessResult,
    compare_logits,
    compare_token_sequences,
    run_correctness_suite,
)
from .profiler import DiffusionProfiler 

__all__ = [
    "LatencyResult",
    "measure_forward_latency",
    "measure_diffusion_step_latency",
    "measure_generation_latency",
    "ThroughputResult",
    "measure_tokens_per_second",
    "measure_steps_per_second",
    "measure_sequences_per_second",
    "CorrectnessResult",
    "compare_logits",
    "compare_token_sequences",
    "run_correctness_suite",
    "DiffusionProfiler",
]