# CP vs non-CP Path Analysis (SM90, H100)

High-density sweep across 40+ configs to validate the CP heuristic.

## Current heuristic (flashinfer 0.6.18)

```python
use_cp = (num_seqs * num_sab_heads * threshold_den < num_sms * threshold_num)
```

Where `threshold_num/den = 1/2` on HBM devices (H100), so:
```python
use_cp = (num_seqs * num_sab_heads < num_sms / 2)  # 66 on H100
```

## Heuristic failures

| seqs | L | par | auto choice | noCP ms | CP ms | actual winner |
|------|------|-----|-------------|---------|-------|---------------|
| 2 | 2048 | 32 | **CP** | 0.281 | 0.408 | noCP |
| 4 | 4096 | 64 | **CP** | 0.208 | 0.296 | noCP |
| 4 | 8192 | 64 | **CP** | 0.370 | 0.500 | noCP |

**Root cause**: heuristic checks only `parallelism`, ignoring `seq_len`.

- Long sequences (L≥4096) incur high CP fixup recurrence cost (chunks × fixup_overhead)
- At seqs=4, L=2048: CP wins (0.191ms vs 0.204ms)
- At seqs=4, L=4096: CP loses (0.296ms vs 0.208ms)

## Correct switching logic

CP is optimal when **both** conditions hold:
1. `par < num_sms / 2` (low parallelism)
2. `max_seq_len <= 2048` (short sequence)

Proposed fix for sglang `gdn_flashinfer.py`:
```python
max_seq_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
use_cp = "auto" if max_seq_len <= 2048 else False
```

This eliminates all 3 WRONG cases while preserving CP wins at:
- seqs=1, L≤16384: 1.88× faster (0.721ms → 0.273ms at L=16384)
- seqs≤4, L≤2048: 5-10% faster

## Full sweep results

See [cp_sweep_full.txt](cp_sweep_full.txt) for all 40+ configs.
