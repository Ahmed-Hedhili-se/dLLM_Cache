# MoE Expert Dispatch Lab

A small, self-contained project to turn scattered exploration (CUDA glossary,
torch.compile, nsys/ncu with Salem, "Tensor Economics: MoE Inference") into
one coherent profile-and-optimize cycle on a toy MoE layer — before doing the
same thing for real on LLaDA-MoE in Phase 3/4.

This is NOT LLaDA. It's a minimal top-k MoE FFN block (router + N expert
FFNs), small enough to fully understand, but structurally identical to the
part of LLaDA-MoE that Phase 4 will optimize with grouped GEMM.

## Files

| File | Step | What it does | Where to run |
|---|---|---|---|
| `moe_naive.py` | 1 | Naive top-k MoE, per-expert python-loop dispatch | anywhere (CPU ok) |
| `moe_compiled.py` | 2 | `torch.compile` wrap + eager-vs-compiled comparison | anywhere, but GPU numbers only matter on H200 |
| `profile_run.py` | 3 & 4 | Thin driver script for `nsys`/`ncu` to attach to | **H200 only** |
| `tensor_economics.py` | 5 | By-hand roofline analysis (FLOPs, bytes, AI, ridge point) | anywhere (pure math) |

Steps 1, 2, and 5 have already been run and verified correct on CPU in this
environment (see results below). Steps 3 and 4 need an actual GPU — run them
on the H200 with Salem's nsys/ncu setup.

## How to run each step

### Step 1 — Baseline
```bash
python moe_naive.py
```
Confirms the layer is correct and gives a naive wall-clock number.

### Step 2 — torch.compile
```bash
python moe_compiled.py
# to see graph breaks:
TORCH_LOGS="graph_breaks" python moe_compiled.py
```
Watch for breaks around the per-expert loop (`nonzero()`, `index_add_`) —
these are the dynamic-shape ops that resist fusion. This is the same class
of problem you'll hit compiling LLaDA-MoE's routing.

### Step 3 — Nsight Systems (on H200)
```bash
nsys profile -o profiles/naive_moe    python profile_run.py --variant naive    --iters 30
nsys profile -o profiles/compiled_moe python profile_run.py --variant compiled --iters 30
nsys stats profiles/naive_moe.nsys-rep
```
Look for: many small serialized kernel launches per expert, with visible
GPU-idle gaps in the naive timeline (CPU launch overhead dominating actual
compute time).

### Step 4 — Nsight Compute (on H200)
```bash
ncu --set full -o profiles/expert_kernel python profile_run.py --variant naive --iters 1
```
Open in the ncu GUI or `ncu --import profiles/expert_kernel.ncu-rep`. Record
Compute (SM) Throughput % vs Memory Throughput %, and Achieved Occupancy for
one expert's fc1 GEMM. This is where Step 5's prediction gets checked against
a real measured number.

### Step 5 — Tensor economics
```bash
python tensor_economics.py
```
Already run here — results below.

### Step 6 (stretch) — Grouped GEMM
Not yet implemented. Natural next step once Steps 1-5 are done: replace the
per-expert python loop in `moe_naive.py` with a single grouped/batched GEMM
(pad each expert's token count to a common size, or use `torch._grouped_mm`
/ a hand-rolled Triton kernel), then re-run Steps 3-5 and compare.

## Results so far (CPU sanity run, d_model=1024, d_hidden=4096, n_experts=8, top_k=2)

- **Step 1**: naive forward correct, ~535ms / 3,825 tok/s on CPU (meaningless
  in absolute terms — CPU, no batching benefit — but confirms the dispatch
  logic gathers/scatters correctly).
- **Step 2**: `torch.compile` gave a modest ~1.06x on CPU; expect a much
  bigger, more interesting gap on the H200 where kernel-launch overhead and
  fusion actually matter. Re-run there.
- **Step 5**: H200 ridge point ≈ **206 FLOPs/byte**. The expert FFN is
  memory-bound below ~1024 tokens/expert and only crosses into compute-bound
  territory above that. At top_k=2 over 8 experts, a batch needs roughly
  **4,000+ concurrent tokens** before the average expert sees enough traffic
  to stop being memory-bound. This is the quantitative version of "grouped
  GEMM matters more as concurrency drops" — directly relevant to your
  concurrency-1-to-64 throughput sweep gate.

## How this connects back to the LLaDA-MoE project

- **Step 2 (torch.compile limits)** previews the graph-break problems you'll
  hit trying to compile LLaDA-MoE's `trust_remote_code` routing in Phase 3.
- **Steps 3-4 (nsys/ncu)** are the exact profiling workflow you'll use to
  diagnose where LLaDA-MoE's serving loop is bottlenecked once you move past
  correctness gating into the throughput-per-concurrency-level optimization.
- **Step 5 (tensor economics)** is the same roofline argument dInfer and the
  "Tensor Economics" article make for why block-wise KV caching and grouped
  GEMM matter for MoE specifically — you've now derived it yourself instead
  of just reading it.
- **Step 6 (grouped GEMM stretch)** is a low-stakes rehearsal for Phase 4's
  grouped GEMM MoE routing work, done on a toy layer you fully understand
  before touching the real model.
