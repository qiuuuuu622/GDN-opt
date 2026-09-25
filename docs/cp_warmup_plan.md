# Gate-Aware CP for GDN Prefill — borrowable from FlashQLA

## The two problems FlashQLA names

GDN chunked prefill has a structural tension:

1. **Per-step kernels are memory-bound** — intermediates round-trip through HBM.
2. **The recurrence kernel's parallelism is only `batch × num_heads`** — on TP
   or small-batch/long-sequence serving, SM occupancy collapses.

Fully fusing solves (1) but worsens (2). FlashQLA's answer: split into **two
fused kernels** with the CP preprocessing in between, and make the CP
preprocessing *gate-aware*.

## What FlashInfer SM90 does today

`chunk_gated_delta_rule` dispatches on `should_use_cp_host`, which considers
**only parallelism** (`par * 2 < num_sms`). No sequence-length term, no gate term.

The CP path runs **four kernels**:

| stage | role |
|---|---|
| `TPrecompute` | KKT + `CollectiveInverse` HMMA 64×64 inverse |
| `MNPrecompute` | per-chunk transfer `M` and state `N` |
| `Fixup` | `S_c1 = S_init @ M + N` |
| `Prefill` | local recurrence per CP chunk |

The `Fixup` relation is explicit in the source (`delta_rule_cp_sm90.py:1997`).

**This is exactly the design FlashQLA argues against** for decaying heads: the
transfer-matrix machinery is O(chunks), and building `M` costs more than the
matmuls it corrects.

## Measured prize

### CP loses at long sequences — the regime production lives in

Production GVA layout (TP=4 ⇒ per-rank q/k=4, v=16), full adapter I/O:

| B | T/seq | par | non-CP ms | CP ms | winner |
|---|---|---|---|---|---|
| 1 | 4096 | 16 | 0.496 | 0.274 | CP 1.81× |
| 1 | 8192 | 16 | 0.806 | 0.312 | CP 2.58× |
| 1 | 16384 | 16 | 1.339 | 0.426 | CP 3.14× |
| 1 | 32768 | 16 | 2.688 | 0.785 | CP 3.42× |
| 4 | 4096 | 64 | 0.394 | 0.502 | non-CP 1.27× |
| 8 | 2048 | 128 | 0.253 | 0.573 | non-CP 2.27× |
| 16 | 1024 | 256 | 0.236 | 0.810 | non-CP 3.43× |

### CP stage cost grows with sequence length

Per-call GPU time (µs), B=1, decaying gates:

| T/seq | Prefill (main) | stage 2 | stage 3 | stage 4 |
|---|---|---|---|---|
| 8192 | 63.3 | 61.1 | 37.0 | 25.1 |
| 16384 | 322.5 | 102.2 | 46.1 | 36.5 |
| 32768 | 575.9 | 197.5 | 83.8 | 34.6 |

Stages 2–4 total **123 µs at T=8192 → 316 µs at T=32768** — that is the
M-matrix machinery, and it scales with chunk count.

### The real gates decay fast enough

Measured from `/prefill_model` weights (36 layers × 64 heads = 2304 heads):

```
per-token alpha: median 0.606

warmup chunks to attenuate an initial-state perturbation below 1e-5:
  p50 = 0.36    p90 = 1.12    p95 = 3.54
  heads needing <= 8 chunks : 96.5%     (FlashQLA claims 60-80%)
  heads needing  > 64 chunks:  2.3%
  heads that never decay    :  0
```

Caveat: computed from `exp(A_log) × softplus(dt_bias)`, i.e. a lower bound on
decay rate. Real activations must be confirmed on trace data before trusting it.

## What to borrow

**1. Warmup CP path (highest value).**
For decaying heads, skip `TPrecompute`/`MNPrecompute`/`Fixup`: each CP rank runs
a short zero-state warmup, writes its local `S_init`, then recurs normally. The
warmup length comes from a separate cheap kernel that scans the gate. Targets
the 123–316 µs of stages 2–4 and, more importantly, makes CP correct-and-fast at
the long sequence lengths production actually serves.

**2. Gate- and length-aware CP selection.**
Replace `par * 2 < num_sms` with a predicate that also carries `seq_len` and the
fraction of non-decaying heads. FlashQLA uses `bs×heads ≤ 40`, or `≤ 56` with
`seq_len ≥ 8192`. Our table shows the crossover moving with length.

**3. v_head_dim splitting** for extra parallelism (2–4×), at the cost of
redundant Q/K traffic. Relevant because TP=4 leaves only `sab=16` units.

**4. Two-kernel split** instead of quad-kernel CP or a single fully-fused kernel.

## Lessons from the in-flight FlashQLA work on h100-8

`/mnt/workspace/inference/flashqla-pool-opt-20260920/` already explored this and
hit real traps — reuse them rather than rediscovering:

- **Correctness**: CP initial state must be indexed **per CP segment (`bb`)**,
  while final state writes back on the **original sequence index**. Conflating
  them was the root cause of RRMSE 0.137. Fix: separate initial-state address
  from final-state address; keep the assertion forbidding pool+CP.
- **TileLang 0.1.12 trap**: `correct_h0` with `T.Pipelined` produced ~30%
  first-boundary error on V-first state. `T.serial` fixed it (0.30 → 0.0043),
  matching an independent torch fixup exactly.
- **Numerical**: A and Ag `BF16` roundings both matter — fixing only one leaves
  max error at 0.2651. Keeping both in FP32 passes (0.0532).
- **All their timing was withdrawn** — another SGLang process shared GPU0 during
  the runs. Any re-measurement must be on an isolated GPU with an audit record.
- Don't open the existing pool entry's `auto_cp`: source forbids pool-indexed +
  `cp_seq_map` combination, and the old Auto-CP preprocessing includes host-side
  `tolist`/chunk construction.

## Proposed next steps

1. Capture the **real gate distribution** from a production trace (not weights):
   fraction of heads whose cumulative log-gate reaches the warmup threshold
   within N chunks. This sets the fallback-head ratio and therefore the prize.
2. Per-stage timing of the four CP kernels **on an isolated GPU**, with the
   interference audit the previous round lacked.
3. Implement warmup CP behind a flag, validated against sequential states with a
   zero-tolerance elementwise check across decaying/mixed/weak gates.
4. Only then re-tune the CP selection predicate.
