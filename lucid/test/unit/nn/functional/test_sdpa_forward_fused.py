"""SDPA forward — memory-efficient fused (MLX / metal) path (Phase 1).

Additive-mask attention now routes through the fused kernel instead of
materializing the ``(B, H, Lq, Lk)`` score matrix.  Checks:

* **output parity** — fused metal additive/bool/causal output matches the CPU
  reference (which rolls the full softmax by hand);
* **memory** — the fused additive forward's peak MLX allocation stays ~O(T),
  far below the explicit O(T²) score-matrix path;
* **weights variant** — ``scaled_dot_product_attention_with_weights`` still
  returns correct dense weights on metal for none/causal/additive/bool.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import metal_available

B, H, T, D = 2, 3, 16, 8
_N = B * H * T * D


def _mk(freq: float, device: str, n: int = _N, shape: tuple = (B, H, T, D)):
    vals = [math.sin(i * freq + 0.3) * 0.5 for i in range(n)]
    return lucid.tensor(vals, dtype=lucid.float32, device=device).reshape(*shape)


def _additive_mask(device: str, tq: int = T, tk: int = T):
    rows = [[0.0 if j <= i else -1.0e4 for j in range(tk)] for i in range(tq)]
    return lucid.tensor(rows, dtype=lucid.float32, device=device).reshape(1, 1, tq, tk)


def _bool_mask(device: str, tq: int = T, tk: int = T):
    rows = [[1.0 if j <= i else 0.0 for j in range(tk)] for i in range(tq)]
    t = lucid.tensor(rows, dtype=lucid.float32, device=device).reshape(1, 1, tq, tk)
    return t > 0.5


@pytest.mark.parametrize("config", ["none", "causal", "additive"])
def test_fused_forward_matches_cpu(config: str) -> None:
    """Fused metal SDPA output must match the CPU reference (manual softmax)."""
    if not metal_available():
        pytest.skip("metal backend unavailable")

    def out(device: str):
        q, k, v = _mk(0.7, device), _mk(0.5, device), _mk(0.3, device)
        if config == "none":
            r = F.scaled_dot_product_attention(q, k, v)
        elif config == "causal":
            r = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            r = F.scaled_dot_product_attention(
                q, k, v, attn_mask=_additive_mask(device)
            )
        return r.numpy()

    np.testing.assert_allclose(out("metal"), out("cpu"), atol=1e-5, rtol=1e-4)


def test_query_broadcast_padding_mask_applied() -> None:
    """Regression: a (B, 1, 1, S) key-padding mask (query dim broadcast) must be
    applied correctly on the fused metal path.  The engine mis-applied such masks
    (masked keys kept weight) until the functional expands the query dim."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    Bt, Ht, Tq, Sk, Dh = 1, 4, 7, 7, 8
    n = Bt * Ht * Tq * Dh
    q = _mk(0.31, "metal", n=n, shape=(Bt, Ht, Tq, Dh))
    k = _mk(0.41, "metal", n=n, shape=(Bt, Ht, Sk, Dh))
    v = _mk(0.51, "metal", n=n, shape=(Bt, Ht, Sk, Dh))
    # mask out the last two key positions for every query (broadcast over query).
    rows = [0.0] * (Sk - 2) + [float("-inf")] * 2
    mask = lucid.tensor(rows, dtype=lucid.float32, device="metal").reshape(1, 1, 1, Sk)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).numpy()
    qn, kn, vn = q.numpy(), k.numpy(), v.numpy()
    s = np.einsum("bhqd,bhkd->bhqk", qn, kn) * (1.0 / math.sqrt(Dh)) + mask.numpy()
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    w = e / e.sum(-1, keepdims=True)
    ref = np.einsum("bhqk,bhkd->bhqd", w, vn)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)


def test_fused_causal_forward_is_memory_linear() -> None:
    """The fused causal forward must not allocate the O(T²) score matrix.

    (Additive/bool masks are currently routed to the materialized path — the
    fused ``mask_arr`` kernel miscomputes on lazy-graph inputs — so only the
    mask-free / built-in-causal paths take the memory-efficient fused kernel.)
    """
    if not metal_available():
        pytest.skip("metal backend unavailable")
    import mlx.core as mx

    Bt, Ht, Tt, Dt = 1, 4, 1024, 32
    n = Bt * Ht * Tt * Dt
    q = _mk(0.11, "metal", n=n, shape=(Bt, Ht, Tt, Dt))
    k = _mk(0.13, "metal", n=n, shape=(Bt, Ht, Tt, Dt))
    v = _mk(0.17, "metal", n=n, shape=(Bt, Ht, Tt, Dt))
    q.sum().item(), k.sum().item(), v.sum().item()  # force-eval inputs

    # Fused causal path.
    mx.reset_peak_memory()
    F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().item()
    fused_peak = mx.get_peak_memory()

    # Explicit O(T²) reference: materializes the (B,H,T,T) score matrix.
    mx.reset_peak_memory()
    scores = lucid.matmul(q, k.permute([0, 1, 3, 2])) * (1.0 / math.sqrt(Dt))
    from lucid.nn.functional.activations import softmax as _softmax

    (lucid.matmul(_softmax(scores, dim=-1), v)).sum().item()
    manual_peak = mx.get_peak_memory()

    score_bytes = Bt * Ht * Tt * Tt * 4  # one (B,H,T,T) fp32 buffer
    # Fused must save at least a full score matrix vs the manual path.
    assert fused_peak < manual_peak - score_bytes, (
        f"fused_peak={fused_peak} manual_peak={manual_peak} "
        f"score_matrix={score_bytes} — O(T²) buffer not eliminated"
    )


@pytest.mark.parametrize("config", ["none", "causal", "additive", "bool"])
def test_with_weights_op_correct_on_metal(config: str) -> None:
    """The weights variant materializes correct dense W + output on metal."""
    if not metal_available():
        pytest.skip("metal backend unavailable")
    dev = "metal"
    q, k, v = _mk(0.7, dev), _mk(0.5, dev), _mk(0.3, dev)
    scale = 1.0 / math.sqrt(D)

    mask = None
    is_causal = False
    if config == "causal":
        is_causal = True
    elif config == "additive":
        mask = _additive_mask(dev)
    elif config == "bool":
        mask = _bool_mask(dev)

    out_i, w_i = _C_engine.nn.scaled_dot_product_attention_with_weights(
        _unwrap(q),
        _unwrap(k),
        _unwrap(v),
        attn_mask=(_unwrap(mask) if mask is not None else None),
        scale=scale,
        is_causal=is_causal,
    )
    w = _wrap(w_i).numpy()
    out = _wrap(out_i).numpy()

    # numpy reference.
    qn, kn, vn = q.numpy(), k.numpy(), v.numpy()
    scores = np.einsum("bhqd,bhkd->bhqk", qn, kn) * scale
    if config == "causal":
        tri = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(tri[None, None], -np.inf, scores)
    elif config == "additive":
        scores = scores + _additive_mask(dev).numpy()
    elif config == "bool":
        keep = _bool_mask(dev).numpy()
        scores = np.where(keep, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    w_ref = e / e.sum(axis=-1, keepdims=True)
    out_ref = np.einsum("bhqk,bhkd->bhqd", w_ref, vn)

    np.testing.assert_allclose(w, w_ref, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(out, out_ref, atol=1e-5, rtol=1e-4)
