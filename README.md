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

## Results (H100, SM90, BF16, heads=16, D=128)

| config | original (2× norm) | optimized (fused) | speedup (norm only) |
|---|---|---|---|
| 1×8192 | 0.183 ms | 0.098 ms | 1.87× |
| 2×4096 | 0.195 ms | 0.098 ms | 1.99× |
| 4×4096 | 0.409 ms | 0.193 ms | 2.12× |
| 8×2048 | 0.404 ms | 0.193 ms | 2.10× |
| 16×1024 | 0.403 ms | 0.179 ms | 2.26× |

Correctness: max absolute diff = 0.00 (bit-identical at BF16) across all configs.

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
