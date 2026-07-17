"""
Test suite for MoEBlock caching correctness.

Checks:
  1. _compute_moe is token-independent: full run == split run (trivial sanity).
  2. Cached FULL update (is_response_full_update=True) matches a no-cache baseline.
  3. Cached PARTIAL update scatters the correct tokens while leaving others unchanged.

Run from the project root:
    python dLLM_Cache_me/test_moe.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.moe import MoEBlock
from models.utils import H
from cache.moe_cache import MoECache
from cache.cache_manager import CacheManager

torch.manual_seed(42)

B, T, prompt_len = 1, 256, 128
response_len = T - prompt_len
moe = MoEBlock()

# Test 1: _compute_moe is token-independent
print("=" * 60)
print("Test 1: _compute_moe full == split")
x = torch.randn(B, T, H)
out_full  = moe._compute_moe(x)
out_split = torch.cat([moe._compute_moe(x[:, :prompt_len]),
                        moe._compute_moe(x[:, prompt_len:])], dim=1)
diff1 = (out_full - out_split).abs().max().item()
print(f"  Max diff (expect 0): {diff1:.2e}")
assert diff1 == 0.0, f"FAIL: {diff1}"
print("  PASS")

# Test 2: Full cached update == no-cache baseline
print("\nTest 2: cached full update == baseline")
cache_mgr = CacheManager(k_p=50, k_r=7, total_steps=128)
cache_p = MoECache()
x0 = torch.randn(B, T, H)
x1 = torch.randn(B, T, H)

k0 = 128  # initial step
moe(x0, cache=cache_p, cache_manager=cache_mgr, k_step=k0, prompt_len=prompt_len)

k7 = 128 - 7  # = 121, step=7, 7%7==0 -> full response refresh
assert cache_mgr.is_response_full_update(k7), f"Expected full update at k={k7}"
out_cached = moe(x1, cache=cache_p, cache_manager=cache_mgr, k_step=k7, prompt_len=prompt_len)

# Prompt NOT refreshed at step 7 (7 % 50 != 0): should match init-step cached value
out_p_expected = moe._compute_moe(x0[:, :prompt_len])
diff2_prompt = (out_cached[:, :prompt_len] - out_p_expected).abs().max().item()
print(f"  Prompt == init cached value:  max diff = {diff2_prompt:.2e}  (expect 0)")

# Response freshly computed on x1
out_r_expected = moe._compute_moe(x1[:, prompt_len:])
diff2_resp = (out_cached[:, prompt_len:] - out_r_expected).abs().max().item()
print(f"  Response == fresh compute:    max diff = {diff2_resp:.2e}  (expect 0)")

assert diff2_prompt == 0.0, f"FAIL prompt: {diff2_prompt}"
assert diff2_resp == 0.0, f"FAIL response: {diff2_resp}"
print("  PASS")

# Test 3: Partial update only modifies selected tokens
print("\nTest 3: cached partial update")
k_partial = 127  # step=1, not a full-response refresh
assert not cache_mgr.is_response_full_update(k_partial)
x2 = torch.randn(B, T, H)

num_update = int(0.25 * response_len)
indices_resp = torch.arange(num_update).unsqueeze(0)  # [B, M] relative to response
update_indices_abs = indices_resp + prompt_len         # absolute into full sequence

resp_before = cache_p.get_response().clone()
moe(x2, cache=cache_p, cache_manager=cache_mgr,
    k_step=k_partial, prompt_len=prompt_len, update_indices=update_indices_abs)
resp_after = cache_p.get_response()

x2_selected = torch.index_select(x2, 1, update_indices_abs.squeeze(0))
out_fresh_selected = moe._compute_moe(x2_selected)
diff3_updated = (resp_after[:, :num_update] - out_fresh_selected).abs().max().item()
print(f"  Updated positions match fresh: max diff = {diff3_updated:.2e}  (expect 0)")

diff3_unchanged = (resp_after[:, num_update:] - resp_before[:, num_update:]).abs().max().item()
print(f"  Untouched positions unchanged: max diff = {diff3_unchanged:.2e}  (expect 0)")

assert diff3_updated == 0.0, f"FAIL updated: {diff3_updated}"
assert diff3_unchanged == 0.0, f"FAIL unchanged: {diff3_unchanged}"
print("  PASS")

print("\n" + "=" * 60)
print("All MoE cache tests passed.")
