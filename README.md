# GDN Prefill Optimization — SM90 (H100)

Optimization target: `gdn_prefill_qkv_prepare_fwd` in sglang 0.5.20 on H100 (SM90).

## Background

profiling on SM90 shows the L2-norm pre-pass dominates GDN prefill time:

```
l2norm (Triton, 2× calls):  0.759 ms   76.8%
GDN kernel (non-CP):        0.218 ms   22.1%
index_copy_ scatter:        0.011 ms    1.1%
```

`gdn_prefill_qkv_prepare_fwd` (sglang `kernels/ops/attention/fla/l2norm.py`) does:
1. `l2norm_fwd(q)` — separate Triton kernel launch
2. `l2norm_fwd(k)` — separate Triton kernel launch
3. return `(q_normed, k_normed, v)` unchanged

Two independent kernel launches means twice the launch overhead and two passes
over HBM for Q+K. The existing kernel also has autotune commented out (fixed
`BT=16, num_warps=8`).

## Optimization

`kernels/gdn_l2norm_fused.py` provides:

- `l2norm_fwd_qk_kernel` — fused Triton kernel that norms Q and K in a single
  launch, autotuned over `BT ∈ {16,32,64}` × `num_warps ∈ {4,8}` ×
  `num_stages ∈ {2,3}`.
- `gdn_prefill_qkv_prepare_fwd_opt` — drop-in replacement for
  `gdn_prefill_qkv_prepare_fwd` using the fused kernel.

## Results

### Fused QK L2-norm (H100, SM90, BF16, D=128)

Speedup varies with `T_eff` (total_tokens × num_heads):

| T_eff range | speedup | typical config |
|---|---|---|
| ≤ 65k | **1.18–1.32×** | 1~4 seqs, L≤2048, H=16 |
| 130k–262k | **1.03–1.20×** | 8+ seqs, L≥4096, H=16 |
| ≥ 524k | **1.01–1.02×** | 16+ seqs, large batch |

Key findings:
- Small batch / short seq: fused kernel wins more (launch overhead dominates)
- Large batch: memory bandwidth saturated, speedup converges to ~1.01×
- GQA/GVA (H≠HV): speedup matches symmetric configs at same T_eff
- Best application: prefill-heavy workloads with batch ≤ 8

Correctness: max absolute diff = 0.00 (bit-identical at BF16) across 30+ shape configs.

## How to apply

Option A — monkey-patch at runtime (no source edit):
```python
import sglang.kernels.ops.attention.fla.l2norm as _l2
from gdn_l2norm_fused import gdn_prefill_qkv_prepare_fwd_opt
_l2.gdn_prefill_qkv_prepare_fwd = gdn_prefill_qkv_prepare_fwd_opt
```

Option B — apply `patch/gdn_l2norm_fused.patch` to sglang source:
```
cd /sgl-workspace/sglang
git apply /path/to/GDN-opt/patch/gdn_l2norm_fused.patch
```

## Files

```
kernels/gdn_l2norm_fused.py   optimized kernel + drop-in wrapper
patch/gdn_l2norm_fused.patch  minimal unified diff for sglang source
tests/test_gdn_l2norm.py      correctness + benchmark tests
```
