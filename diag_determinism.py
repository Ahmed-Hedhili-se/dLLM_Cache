"""
Diagnostic: Is the logit difference coming from CUDA non-determinism
in F.scaled_dot_product_attention, or from an actual code-path difference?

Test: Call the same model(input_ids) with NO caches twice.
If the logits differ, the problem is CUDA non-determinism, not our code.
"""
import torch
import sys
sys.path.insert(0, '.')

from models.model import LLaDAMoESmall
from models.utils import MASK_ID

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LLaDAMoESmall().to(torch.bfloat16).to(device).eval()

# Load weights if available
import os
if os.path.isdir('weights'):
    from models.loader import load_weights
    load_weights(model, 'weights')
    print("Weights loaded.")
else:
    print("Using random weights (no weights/ dir found).")

input_ids = torch.randint(0, 1000, (1, 256), device=device)
input_ids[0, :128] = torch.arange(1, 129)
input_ids[0, 128:] = MASK_ID

print("\n=== Test 1: Two identical baseline calls (no caches) ===")
with torch.no_grad():
    logits_a = model(input_ids).clone()
    logits_b = model(input_ids).clone()

diff = (logits_a - logits_b).abs()
print(f"  max_abs_diff = {diff.max().item():.6e}")
print(f"  mean_abs_diff = {diff.mean().item():.6e}")
if diff.max().item() == 0:
    print("  → DETERMINISTIC: two baseline calls produce identical logits")
else:
    print("  → NON-DETERMINISTIC: even the baseline differs across calls!")
    print("    This means the logits FAIL is caused by CUDA non-determinism,")
    print("    NOT by a code bug in the caching implementation.")

print("\n=== Test 2: Baseline vs cached initial step ===")
from cache.cache_manager import CacheManager
from cache.embedding_cache import EmbeddingCache
from cache.attention_cache import AttentionCache
from cache.base_cache import BaseCache

cache_manager = CacheManager(k_p=50, k_r=7, total_steps=128)
caches = {
    'embed': EmbeddingCache(),
    'layers': [
        {'attn': AttentionCache(), 'mlp': BaseCache()}
        for _ in range(16)
    ]
}

with torch.no_grad():
    baseline = model(input_ids).clone()
    cached = model(
        input_ids,
        cache_manager=cache_manager,
        caches=caches,
        k_step=128,  # initial step
        prompt_len=128,
        update_ratio=1.0
    ).clone()

diff2 = (baseline - cached).abs()
print(f"  max_abs_diff = {diff2.max().item():.6e}")
print(f"  mean_abs_diff = {diff2.mean().item():.6e}")
if diff2.max().item() == 0:
    print("  → EXACT MATCH: cached initial step is byte-identical to baseline")
elif diff2.max().item() < 1e-2:
    print("  → CLOSE MATCH: tiny bfloat16 rounding, not a real divergence")
else:
    print("  → DIVERGENCE: there's a real code-path difference")

print("\n=== Test 3: Enable deterministic CUDA ===")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)  # May fail on some ops

with torch.no_grad():
    logits_c = model(input_ids).clone()
    logits_d = model(input_ids).clone()

diff3 = (logits_c - logits_d).abs()
print(f"  max_abs_diff = {diff3.max().item():.6e}")
print(f"  mean_abs_diff = {diff3.mean().item():.6e}")
if diff3.max().item() == 0:
    print("  → DETERMINISTIC mode: identical logits")
else:
    print("  → Still non-deterministic even with cudnn.deterministic=True")
    print("    The SDPA kernel itself may be non-deterministic.")
