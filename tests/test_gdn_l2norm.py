"""
Correctness and benchmark tests for the fused GDN L2-norm kernel.

Run on h100-8:
    CUDA_VISIBLE_DEVICES=0 python tests/test_gdn_l2norm.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
sys.path.insert(0, "/sgl-workspace/sglang/python")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from kernels.gdn_l2norm_fused import gdn_prefill_qkv_prepare_fwd_opt

WARMUP = 5
REPS = 100


def time_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / REPS


# ── correctness ───────────────────────────────────────────────────────────────

CORRECTNESS_CONFIGS = [
    (1 * 4096, 16, 16, 128),   # 1 seq × 4096 tokens
    (4 * 4096, 16, 16, 128),   # 4 seqs × 4096 tokens
    (8 * 1024, 16, 16, 128),   # 8 seqs × 1024 tokens
    (4 * 1024, 32, 16, 128),   # GQA 2:1
    (4 * 1024, 16, 32, 128),   # GVA 1:2
]


def test_correctness():
    all_pass = True
    print("=== correctness ===")
    print(f"{'config':>35} | {'q maxdiff':>10} {'k maxdiff':>10} | result")
    print("-" * 75)
    for (T, H_QK, H_V, D) in CORRECTNESS_CONFIGS:
        torch.manual_seed(T + H_QK)
        q = torch.randn(T, H_QK, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(T, H_QK, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(T, H_V,  D, device="cuda", dtype=torch.bfloat16)

        # reference: original two-call path
        q_ref = l2norm_fwd(q.clone())
        k_ref = l2norm_fwd(k.clone())

        # optimized fused path
        q_opt, k_opt, v_opt = gdn_prefill_qkv_prepare_fwd_opt(q.clone(), k.clone(), v.clone())

        q_diff = (q_ref.float() - q_opt.float()).abs().max().item()
        k_diff = (k_ref.float() - k_opt.float()).abs().max().item()
        v_diff = (v.float() - v_opt.float()).abs().max().item()
        assert v_diff == 0.0, f"V was modified: max diff {v_diff}"

        # bit-identical expected since both use float32 accumulation over same data
        passed = q_diff == 0.0 and k_diff == 0.0
        all_pass = all_pass and passed
        label = f"T={T} H_QK={H_QK} H_V={H_V} D={D}"
        print(f"{label:>35} | {q_diff:>10.2e} {k_diff:>10.2e} | {'PASS' if passed else 'FAIL'}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    assert all_pass, "correctness check failed"


# ── benchmark ─────────────────────────────────────────────────────────────────

BENCH_CONFIGS = [
    (1, 8192),
    (2, 4096),
    (4, 4096),
    (8, 2048),
    (16, 1024),
    (32, 512),
]


def test_benchmark():
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    H, HV, D = 16, 16, 128

    print("\n=== benchmark ===")
    print(f"{'config':>22} | {'old 2×norm':>11} {'fused norm':>11} {'gdn kernel':>11} {'speedup':>8}")
    print("-" * 73)

    for (num_seqs, seq_len) in BENCH_CONFIGS:
        T = num_seqs * seq_len
        torch.manual_seed(T)
        q = torch.randn(T, H,  D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(T, H,  D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(T, HV, D, device="cuda", dtype=torch.bfloat16)

        # old path: two separate l2norm_fwd calls
        t_old = time_ms(lambda: (l2norm_fwd(q), l2norm_fwd(k)))

        # new path: fused
        t_new = time_ms(lambda: gdn_prefill_qkv_prepare_fwd_opt(q, k, v))

        # GDN kernel timing for context
        A_log = torch.rand(HV, device="cuda").log()
        g = -A_log.exp().view(1, HV) * F.softplus(
            0.01 * torch.ones(T, HV, device="cuda")
        )
        beta = torch.sigmoid(torch.randn(T, HV, device="cuda"))
        cu = torch.tensor(
            [i * seq_len for i in range(num_seqs + 1)],
            device="cuda", dtype=torch.int64,
        )
        init = torch.zeros(num_seqs, HV, D, D, device="cuda")
        q_n, k_n, _ = gdn_prefill_qkv_prepare_fwd_opt(q, k, v)
        kw = dict(
            g=g, beta=beta, scale=1.0 / math.sqrt(D),
            initial_state=init.clone(), output_final_state=True,
            cu_seqlens=cu, use_qk_l2norm_in_kernel=False, use_cp=False,
        )
        t_kernel = time_ms(lambda: chunk_gated_delta_rule(q_n, k_n, v, **kw))

        label = f"seqs={num_seqs} L={seq_len}"
        print(
            f"{label:>22} | {t_old:>10.4f}ms {t_new:>10.4f}ms "
            f"{t_kernel:>10.4f}ms {t_old/t_new:>8.2f}x"
        )


if __name__ == "__main__":
    test_correctness()
    test_benchmark()
