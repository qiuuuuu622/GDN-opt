"""
Diagnose WHY flashinfer_gdn_prefill_default() returns None on a given host.

The upstream function is a chain of 13 silent `return None` gates. This module
re-implements the gate chain verbatim but REPORTS every gate's value instead of
bailing silently, so the offending condition is identified in one run.

Usage (inside the target sglang image, at the point a ModelRunner exists):

    # Option A - drop-in call with a live runner:
    from diag_gdn_prefill_gate import diag
    diag(runner)

    # Option B - monkeypatch so the next backend resolution prints the trace:
    from diag_gdn_prefill_gate import install
    install()          # call before ModelRunner is constructed

Both are read-only: they never change the selection, they only explain it.
"""

from __future__ import annotations

import torch

_SEP = "=" * 78


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return f"<raised {type(exc).__name__}: {exc}>"


def collect(model_runner):
    """Evaluate every gate and return an ordered list of (name, value, verdict)."""
    from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
    from sglang.srt.runtime_context import get_exec, get_memory, get_schedule

    rows = []

    def add(name, value, ok, note=""):
        rows.append((name, value, ok, note))

    mamba = get_exec().mamba
    schedule = get_schedule()
    memory = get_memory()
    exec_ctx = get_exec()

    # --- gate 1: explicit prefill backend -------------------------------
    v = mamba.linear_attn_prefill_backend
    add("linear_attn_prefill_backend", v, v is None,
        "explicit --linear-attn-prefill-backend wins; must be unset for auto")

    # --- gate 2: base backend -------------------------------------------
    v = mamba.linear_attn_backend
    add("linear_attn_backend", v, v == "triton",
        "base must be 'triton' for the auto-default to engage")

    # --- gate 3/4: deterministic + page-major ---------------------------
    v = exec_ctx.deterministic.enable_deterministic_inference
    add("enable_deterministic_inference", v, not v,
        "flashinfer prefill is unsupported under deterministic inference")
    v = memory.enable_page_major_kv_layout
    add("enable_page_major_kv_layout", v, not v,
        "flashinfer prefill is unsupported with page-major KV")

    # --- gate 5/6: arch + cuda ------------------------------------------
    sm_major = _safe(lambda: torch.cuda.get_device_capability()[0], 0)
    cuda_version = torch.version.cuda
    add("sm_major", sm_major, sm_major in (9, 10), "must be 9 (Hopper) or 10 (Blackwell)")
    add("torch.version.cuda", cuda_version,
        not (sm_major == 10 and (cuda_version is None
                                 or int(cuda_version.split(".", 1)[0]) < 13)),
        "SM100 requires CUDA >= 13")

    if sm_major == 10:
        max_chunk, expected_state_dtype = 8192, torch.bfloat16
    else:
        max_chunk, expected_state_dtype = 32768, torch.float32
    add("max_chunk (derived)", max_chunk, True, f"from sm_major={sm_major}")
    add("expected_state_dtype", expected_state_dtype, True, "from sm_major")

    # --- gate 7: dynamic chunking ---------------------------------------
    v = schedule.enable_dynamic_chunking
    add("schedule.enable_dynamic_chunking", v, not v, "dynamic chunking disqualifies")

    # --- gate 8/9: chunk size -------------------------------------------
    # NOTE: this is get_schedule().chunked_prefill_size, NOT the raw server_arg.
    chunk_size = schedule.chunked_prefill_size
    add("get_schedule().chunked_prefill_size", chunk_size, chunk_size is not None,
        "None here means the scheduler never resolved a positive value")
    add("1 <= chunk_size <= max_chunk",
        chunk_size, isinstance(chunk_size, int) and 1 <= chunk_size <= max_chunk,
        f"max_chunk={max_chunk}")

    # --- gate 10/11: head dims ------------------------------------------
    cfg = hybrid_gdn_config(model_runner.model_config)
    add("hybrid_gdn_config", type(cfg).__name__ if cfg is not None else None,
        cfg is not None, "must be a GDN-family config")
    kd = getattr(cfg, "linear_key_head_dim", None) if cfg is not None else None
    vd = getattr(cfg, "linear_value_head_dim", None) if cfg is not None else None
    add("linear_key_head_dim", kd, kd == 128, "must be 128")
    add("linear_value_head_dim", vd, vd == 128, "must be 128")

    # --- gate 12: mamba temporal dtype ----------------------------------
    dt = _safe(lambda: model_runner.req_to_token_pool.mamba_pool
               .mamba_cache.temporal.dtype)
    add("mamba_cache.temporal.dtype", dt, dt == expected_state_dtype,
        f"must equal {expected_state_dtype} on SM{sm_major}0")

    # --- gate 13: kernel availability -----------------------------------
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )
    avail = _safe(is_flashinfer_gdn_prefill_available, "<raised>")
    add("is_flashinfer_gdn_prefill_available()", avail, avail is True,
        "flashinfer.gdn_prefill + SM90 DSL must import")

    return rows


def diag(model_runner):
    """Print the full gate trace and name the first blocking gate."""
    rows = collect(model_runner)
    print(_SEP)
    print("GDN prefill auto-default gate trace")
    print(_SEP)
    first_blocker = None
    width = max(len(r[0]) for r in rows)
    for name, value, ok, note in rows:
        mark = "OK  " if ok else "BLOCK"
        if not ok and first_blocker is None:
            first_blocker = name
        print(f"  [{mark}] {name:<{width}} = {value!r}")
        if not ok:
            print(f"           -> {note}")
    print(_SEP)
    if first_blocker is None:
        print("VERDICT: all gates pass -> prefill should be 'flashinfer'.")
    else:
        print(f"VERDICT: blocked by '{first_blocker}' -> prefill falls back to 'triton'.")
    print(_SEP)
    return first_blocker


def install():
    """Monkeypatch the upstream function to also emit a gate trace.

    Read-only with respect to behavior: whatever the original returns is still
    what gets returned; the trace is printed alongside it.
    """
    import sglang.srt.layers.attention.linear.gdn_backend as gb

    original = gb.flashinfer_gdn_prefill_default
    state = {"runner": None}

    def wrapper(model_runner):
        result = original(model_runner)
        try:
            diag(model_runner)
        except Exception as exc:  # noqa: BLE001
            print(f"[diag] gate trace failed: {type(exc).__name__}: {exc}")
        print(f"[diag] flashinfer_gdn_prefill_default returned: {result!r}")
        return result

    gb.flashinfer_gdn_prefill_default = wrapper
    print("[diag] installed gate-trace wrapper on "
          "flashinfer_gdn_prefill_default")
    return wrapper


if __name__ == "__main__":
    print(__doc__)
    print("This module is a library; import it from a process that has a "
          "ModelRunner, or call install() before ModelRunner construction.")
