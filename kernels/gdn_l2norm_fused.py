"""
Fused L2-norm kernel for GDN prefill on SM90 (H100).

Drop-in replacement for sglang's gdn_prefill_qkv_prepare_fwd that norms
Q and K in a single Triton kernel launch instead of two separate ones.

Usage (monkey-patch):
    import sglang.kernels.ops.attention.fla.l2norm as _l2
    from kernels.gdn_l2norm_fused import gdn_prefill_qkv_prepare_fwd_opt
    _l2.gdn_prefill_qkv_prepare_fwd = gdn_prefill_qkv_prepare_fwd_opt
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=nw, num_stages=ns)
        for BT in [16, 32, 64]
        for nw in [4, 8]
        for ns in [2, 3]
    ],
    key=["T", "D"],
)
@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_qk_kernel(
    q,
    k,
    oq,
    ok,
    eps,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Norm Q and K rows in one kernel, halving launch overhead and HBM passes."""
    i_t = tl.program_id(0)
    p_q = tl.make_block_ptr(q, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_k = tl.make_block_ptr(k, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q, axis=1) + eps)[:, None]
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k, axis=1) + eps)[:, None]
    p_oq = tl.make_block_ptr(oq, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_ok = tl.make_block_ptr(ok, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_oq, b_q.to(p_oq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_ok, b_k.to(p_ok.dtype.element_ty), boundary_check=(0, 1))


def _l2norm_fwd_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-launch L2-norm for Q and K (both contiguous [T*H, D] views)."""
    T_eff = q.numel() // q.shape[-1]
    D = q.shape[-1]
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    BD = min(65536 // q.element_size(), triton.next_power_of_2(D))
    grid = lambda meta: (triton.cdiv(T_eff, meta["BT"]),)
    l2norm_fwd_qk_kernel[grid](
        q.reshape(T_eff, D),
        k.reshape(T_eff, D),
        oq.reshape(T_eff, D),
        ok.reshape(T_eff, D),
        eps,
        T=T_eff,
        D=D,
        BD=BD,
    )
    return oq, ok


def gdn_prefill_qkv_prepare_fwd_opt(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized drop-in for sglang's gdn_prefill_qkv_prepare_fwd.

    For contiguous Q/K [T, H, D]: fuses the two l2norm launches into one.
    For non-contiguous inputs: falls back to the original implementation.
    V is returned unchanged (no norm needed).
    """
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError(
            f"Expected Q/K [T, H, D] and V [T, Hv, D]; got {q.shape=}, {k.shape=}, {v.shape=}"
        )
    if v.shape[0] != q.shape[0] or v.shape[2] != q.shape[2]:
        raise ValueError(f"Mismatched T or D between Q and V: {q.shape=}, {v.shape=}")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, V must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K, V must have the same dtype")

    if q.is_contiguous() and k.is_contiguous():
        q_normed, k_normed = _l2norm_fwd_qk(q, k, eps)
        return q_normed, k_normed, v

    # Non-contiguous fallback: materialize then fuse norm
    q_c = q.contiguous()
    k_c = k.contiguous()
    q_normed, k_normed = _l2norm_fwd_qk(q_c, k_c, eps)
    return q_normed, k_normed, v.contiguous()
