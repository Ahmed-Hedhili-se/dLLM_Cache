
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _check_torch():
    if not _HAS_TORCH:
        raise ImportError("sampling.py requires torch to be installed")


def temperature_scale(logits, temperature: float = 1.0):
    """Applies temperature scaling to logits. temperature <= 0 is a no-op (treated as greedy)."""
    _check_torch()
    if temperature <= 0:
        return logits
    return logits / temperature


def greedy(logits):
    _check_torch()
    return torch.argmax(logits, dim=-1)


def top_k(logits, k: int = 50, temperature: float = 1.0, generator=None):
    _check_torch()
    logits = temperature_scale(logits, temperature)

    k = min(k, logits.size(-1))
    values, indices = torch.topk(logits, k, dim=-1)
    probs = F.softmax(values, dim=-1)

    orig_shape = logits.shape[:-1]
    flat_probs = probs.reshape(-1, k)
    flat_indices = indices.reshape(-1, k)

    sampled = torch.multinomial(flat_probs, num_samples=1, generator=generator)
    chosen = torch.gather(flat_indices, -1, sampled)

    return chosen.reshape(orig_shape)


def top_p(logits, p: float = 0.9, temperature: float = 1.0, generator=None):
    """Nucleus sampling: samples from the smallest token set with cumulative probability >= p."""
    _check_torch()
    logits = temperature_scale(logits, temperature)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Drop tokens once the running total (before adding them) exceeds p,
    # always keeping at least the single most likely token.
    sorted_mask = (cumulative_probs - sorted_probs) > p
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    probs = F.softmax(sorted_logits, dim=-1)

    orig_shape = probs.shape
    flat_probs = probs.reshape(-1, orig_shape[-1])
    flat_indices = sorted_indices.reshape(-1, orig_shape[-1])

    sampled = torch.multinomial(flat_probs, num_samples=1, generator=generator)
    chosen = torch.gather(flat_indices, -1, sampled)

    return chosen.reshape(orig_shape[:-1])


def sample(logits, strategy: str = "greedy", **kwargs):
    if strategy == "greedy":
        return greedy(logits)
    elif strategy == "top_k":
        return top_k(logits, **kwargs)
    elif strategy == "top_p":
        return top_p(logits, **kwargs)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy!r}")


def token_confidence(logits, token_ids):
    _check_torch()
    probs = F.softmax(logits, dim=-1)
    return torch.gather(probs, -1, token_ids.unsqueeze(-1)).squeeze(-1)