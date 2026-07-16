"""
Measures how long different parts of the diffusion pipeline take:
- a single model forward pass
- a single diffusion denoising step
- an entire end-to-end generation call

All timing functions follow the same pattern: warm up a few times
(to avoid measuring one-time costs like CUDA kernel compilation or
lazy initialization), then run several timed repetitions and report
summary statistics rather than a single noisy number.
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class LatencyResult:
    name: str
    samples_ms: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples_ms)

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.samples_ms) if len(self.samples_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)

    def __repr__(self):
        return (
            f"{self.name}: mean={self.mean_ms:.3f}ms  std={self.std_ms:.3f}ms  "
            f"min={self.min_ms:.3f}ms  max={self.max_ms:.3f}ms  (n={len(self.samples_ms)})"
        )
def _sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_fn(fn: Callable, warmup: int, repeats: int) -> List[float]:
    """Runs fn() `warmup` times untimed, then `repeats` times timed. Returns ms per call."""
    for _ in range(warmup):
        fn()
    _sync()

    samples = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        end = time.perf_counter()
        samples.append((end - start) * 1000.0)

    return samples



def measure_forward_latency(
    model_fn: Callable, tokens: list, warmup: int = 3, repeats: int = 10
) -> LatencyResult:
    samples = _time_fn(lambda: model_fn(tokens), warmup, repeats)
    return LatencyResult(name="single_forward", samples_ms=samples)


def measure_diffusion_step_latency(
    generator, tokens: list, step: int, editable: list, warmup: int = 3, repeats: int = 10
) -> LatencyResult:
    def run_once():
        generator._denoise_step(list(tokens), step, editable)

    samples = _time_fn(run_once, warmup, repeats)
    return LatencyResult(name="diffusion_step", samples_ms=samples)



def measure_generation_latency(
    engine, prompt, gen_length: int = 64, warmup: int = 1, repeats: int = 5
) -> LatencyResult:
    samples = _time_fn(
        lambda: engine.generate(prompt, gen_length=gen_length), warmup, repeats
    )
    return LatencyResult(name="total_generation", samples_ms=samples)