# dLLM-Cache

## References

[![dLLM-Cache Paper](https://img.shields.io/badge/arXiv-2506.06295-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.06295)

[![LLaDA Paper](https://img.shields.io/badge/arXiv-2509.24389-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.24389)

[![HuggingFace](https://img.shields.io/badge/HuggingFace-LLaDA--MoE--7B--A1B-yellow?logo=huggingface)](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)

A PyTorch implementation of **dLLM-Cache**...

A PyTorch implementation of **dLLM-Cache** — a training-free KV-cache acceleration framework for **Diffusion Large Language Models (dLLMs)**, applied to a parameter-reduced version of [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-8B-Instruct).

---

## Overview

Unlike autoregressive LLMs, dLLMs (e.g. LLaDA) denoise *all* response tokens simultaneously across hundreds of steps, making naive KV-cache strategies inapplicable. This repository implements the caching strategy from:

> **dLLM-Cache: Accelerating Diffusion Large Language Model Inference via Adaptive Caching**
> [[arXiv:2506.06295]](https://arxiv.org/abs/2506.06295)

The key insight: across consecutive denoising steps, prompt token representations are *nearly static* while most response token representations change only marginally. dLLM-Cache exploits this by caching four tensors per layer — **K**, **V**, **AttnOut** (attention output), and **FFNOut** — and selectively recomputing only the subset of response tokens whose Value vectors have drifted the most, identified via cosine similarity (V-verify, Eq. 7 of the paper).

The base model is a scaled-down version of LLaDA-MoE built from the architecture of:

> **LLaDA: Large Language Diffusion with mAsking**
> [[arXiv:2509.24389]](https://arxiv.org/abs/2509.24389)

---

## Architecture

### Model (`models/`)

A small Mixture-of-Experts masked diffusion transformer, reduced in size from the [HuggingFace checkpoint](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct):

| Hyperparameter | Value |
|---|---|
| Hidden size (`H`) | 512 |
| Attention heads (`NH`) | 8 |
| KV heads (`KVH`) | 8 |
| Head dimension (`HD`) | 64 |
| Layers (`NL`) | 4 |
| Experts (`NE`) | 16 |
| Active experts per token (`TOPK`) | 4 |
| Expert intermediate size (`EI`) | 256 |
| Vocabulary size (`VS`) | 157,184 |
| RoPE base (`THETA`) | 50,000 |

**Files:**

| File | Description |
|---|---|
| `model.py` | Top-level `LLaDAMoESmall` — embedding, RoPE, layer stack, LM head |
| `layer.py` | Single transformer block: `Attention` + `MoEBlock` with residual connections |
| `attention.py` | Multi-head attention with GQA, RMSNorm on Q/K, RoPE, and full cache integration |
| `moe.py` | Top-K sparse MoE block with token routing |
| `expert.py` | Individual expert MLP |
| `embedding.py` | Token embedding lookup |
| `rope.py` | Rotary position embeddings |
| `rmsnorm.py` | Root Mean Square Layer Normalization |
| `utils.py` | Shared hyperparameter constants |

---

### Cache (`cache/`)

Implements the full dLLM-Cache mechanism from Section 3.2 of the paper. Each transformer layer has two cache objects: one `AttentionCache` and one `MoECache`.

```
cache/
├── base_cache.py         # Abstract base: prompt_cache / response_cache split
├── attention_cache.py    # Caches K, V, AttnOut for prompt and response
├── hidden_cache.py       # Caches FFNOut for dense layers
├── moe_cache.py          # Caches FFNOut for MoE layers (same interface)
├── embedding_cache.py    # Caches prompt token embeddings (bonus, not in paper)
├── token_tracker.py      # V-verify: cosine similarity → bottom-ρ token selection
├── selective_compute.py  # gather_tokens / scatter_tokens primitives
├── cache_manager.py      # Scheduling: Kp / Kr interval logic
└── __init__.py
```

**Cache update logic per step `k`:**

| Condition | Prompt cache | Response cache |
|---|---|---|
| `k == K` (initial) | Full recompute + store | Full recompute + store |
| `k % Kp == 0` | Full recompute + store | — |
| `k % Kr == 0` | — | Full recompute + store |
| Otherwise | Read from cache | V-verify → partial scatter |

The four conditions are **evaluated independently**, so `Kp` and `Kr` do not need to be multiples of each other (e.g. the paper's GSM8K setting `Kp=50, Kr=7` works correctly).

---

### Diffusion Engine (`diffusion/`)

| File | Description |
|---|---|
| `scheduler.py` | Noise schedule and step-count management |
| `masking.py` | Mask-token sampling and replacement |
| `sampling.py` | Token sampling strategies (argmax, top-p, etc.) |
| `generator.py` | High-level generation loop |
| `inference.py` | End-to-end inference with cache warm-up |

---

### CUDA Kernels (`cuda/`)

Custom CUDA kernels for the gather/scatter primitives at the core of the partial-update mechanism. The pure-PyTorch fallback in `cache/selective_compute.py` is fully functional; these kernels are drop-in replacements for performance-critical deployments.

| File | Description |
|---|---|
| `gather.cu` | Fused token gather kernel |
| `scatter.cu` | Fused token scatter kernel |
| `attention_kernel.cu` | Selective attention computation |
| `selective_attention.cu` | Partial-sequence attention routing |

---

## How It Works

### V-verify (Token Selection)

At each adaptive step, `TokenTracker.verify()` identifies which response tokens to recompute:

```python
# Eq. 7 from the paper
similarity = F.cosine_similarity(v_current, v_cached, dim=-1)  # [B, T_r]
indices = torch.topk(similarity, k=floor(ρ * T_r), largest=False).indices
```

Tokens with the **lowest cosine similarity** to their cached Values have drifted the most and are selected for full recomputation. The rest are served directly from cache.

### Four-Case Branch Logic

`attention.py` and `moe.py` handle prompt and response independently:

```
Step k
 ├─ is_initial or is_prompt_up?
 │   YES → recompute prompt Q/K/V, run attention, update cache
 │   NO  → read K_p, V_p, AttnOut_p from cache
 │
 └─ is_initial or is_resp_full?
     YES → recompute response Q/K/V, run attention, update cache
     NO  → V-verify → gather active tokens → partial Q/K/V → scatter results
```

---

## Installation

```bash
git clone https://github.com/<your-username>/dLLM-cache
cd dLLM-cache
pip install torch
```

No additional dependencies are required for the pure-PyTorch path. For CUDA kernel compilation:

```bash
cd cuda
python setup.py build_ext --inplace
```

---

## Usage

### Basic forward pass (no cache)

```python
import torch
from models.model import LLaDAMoESmall

model = LLaDAMoESmall().to(torch.bfloat16).cuda()
input_ids = torch.randint(0, 1000, (1, 64)).cuda()
logits = model(input_ids)  # [1, 64, vocab_size]
```

### Forward pass with dLLM-Cache

```python
from cache import (
    CacheManager, EmbeddingCache,
    AttentionCache, MoECache
)

# Kp=50, Kr=10 — prompt refreshed every 50 steps, response every 10
cache_manager = CacheManager(k_p=50, k_r=10, total_steps=256)

caches = {
    'embed': EmbeddingCache(),
    'layers': [
        {'attn': AttentionCache(), 'mlp': MoECache()}
        for _ in range(len(model.layers))
    ]
}

prompt_len = 32  # number of prompt tokens

for k_step in range(256, 0, -1):
    logits = model(
        input_ids,
        cache_manager=cache_manager,
        caches=caches,
        k_step=k_step,
        prompt_len=prompt_len,
        update_ratio=0.25  # ρ: fraction of response tokens to recompute per adaptive step
    )
    # ... mask token sampling / update input_ids ...
```

---

## Tests

```bash
# Smoke test: cache init and multi-step consistency
python -m tests.test_model_cache

# Numerical consistency test
python -m tests.test_numerical
```

---

## Project Structure

```
dLLM-cache/
├── models/               # Transformer model (LLaDAMoESmall)
│   ├── model.py
│   ├── layer.py
│   ├── attention.py
│   ├── moe.py
│   ├── expert.py
│   ├── embedding.py
│   ├── rope.py
│   ├── rmsnorm.py
│   └── utils.py
├── cache/                # dLLM-Cache implementation
│   ├── attention_cache.py
│   ├── base_cache.py
│   ├── cache_manager.py
│   ├── embedding_cache.py
│   ├── hidden_cache.py
│   ├── moe_cache.py
│   ├── selective_compute.py
│   └── token_tracker.py
├── diffusion/            # Masked diffusion generation engine
│   ├── generator.py
│   ├── inference.py
│   ├── masking.py
│   ├── sampling.py
│   └── scheduler.py
├── cuda/                 # Custom CUDA kernels (optional)
│   ├── attention_kernel.cu
│   ├── gather.cu
│   ├── scatter.cu
│   └── selective_attention.cu
├── benchmarks/
├── configs/
├── tests/
│   ├── test_model_cache.py
│   └── test_numerical.py
└── main.py
```




This project is licensed under the **MIT License**.
