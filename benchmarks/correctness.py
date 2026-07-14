from dataclasses import dataclass

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class CorrectnessResult:
    name: str
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    details: str = ""

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: max_abs_diff={self.max_abs_diff:.6e}  "
            f"mean_abs_diff={self.mean_abs_diff:.6e}  {self.details}"
        )




def compare_logits(
    baseline_logits, optimized_logits, rtol: float = 1e-3, atol: float = 1e-5,
    name: str = "logits_comparison",
) -> CorrectnessResult:
    if not _HAS_TORCH:
        raise ImportError("correctness.py requires torch")

    if baseline_logits.shape != optimized_logits.shape:
        return CorrectnessResult(
            name=name, passed=False, max_abs_diff=float("inf"), mean_abs_diff=float("inf"),
            details=f"shape mismatch: {baseline_logits.shape} vs {optimized_logits.shape}",
        )

    diff = (baseline_logits.float() - optimized_logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    passed = torch.allclose(baseline_logits.float(), optimized_logits.float(), rtol=rtol, atol=atol)

    return CorrectnessResult(name=name, passed=passed, max_abs_diff=max_diff, mean_abs_diff=mean_diff)




def compare_token_sequences(
    baseline_tokens: list, optimized_tokens: list, name: str = "token_sequence_comparison"
) -> CorrectnessResult:
    length_mismatch = len(baseline_tokens) != len(optimized_tokens)
    mismatches = sum(1 for a, b in zip(baseline_tokens, optimized_tokens) if a != b)

    passed = (mismatches == 0) and not length_mismatch
    details = f"{mismatches} token mismatches"
    if length_mismatch:
        details += f", length mismatch: {len(baseline_tokens)} vs {len(optimized_tokens)}"

    return CorrectnessResult(
        name=name, passed=passed, max_abs_diff=float(mismatches), mean_abs_diff=float(mismatches),
        details=details,
    )




def run_correctness_suite(
    baseline_fn, optimized_fn, inputs: list, rtol: float = 1e-3, atol: float = 1e-5
) -> list:
    results = []
    for i, item in enumerate(inputs):
        baseline_out = baseline_fn(item)
        optimized_out = optimized_fn(item)
        result = compare_logits(baseline_out, optimized_out, rtol=rtol, atol=atol, name=f"input_{i}")
        results.append(result)
    return results