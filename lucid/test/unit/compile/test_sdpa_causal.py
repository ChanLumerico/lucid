"""Compiled scaled-dot-product-attention — causal path + mask safety.

``is_causal=True`` used to force eager fallback in the MPSGraph emitter.
It now emits a lower-triangular −∞ additive mask before the softmax, so
autoregressive decoders built on the fused SDPA op (``nn.MultiheadAttention``
/ ``nn.TransformerDecoderLayer`` with ``is_causal=True`` and no Python-built
mask) compile into a single executable.

Two correctness properties are pinned here:

  1. **Causal compiles, parity-exact vs eager** — square self-attention and
     the non-square (cached-decode-shaped, Lq < Lk) case, which must use the
     bottom-right alignment ``j <= i + (Lk - Lq)``.
  2. **A declared additive mask never compiles to a wrong result** — the
     tracer does not wire the auxiliary mask tensor into the graph, so the
     emitter falls back to eager rather than silently dropping the mask
     (this also keeps the ``is_causal`` + mask combination correct, since
     the eager backend lets the additive mask win over ``is_causal``).
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


def _qkv(B: int, H: int, Lq: int, Lk: int, D: int) -> tuple[lucid.Tensor, ...]:
    return (
        metal_tensor(B, H, Lq, D),
        metal_tensor(B, H, Lk, D),
        metal_tensor(B, H, Lk, D),
    )


class _Causal(nn.Module):
    def forward(self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class _Plain(nn.Module):
    def forward(self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


class _MaskOnly(nn.Module):
    def forward(
        self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor, m: lucid.Tensor
    ) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m)


class _CausalPlusMask(nn.Module):
    def forward(
        self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor, m: lucid.Tensor
    ) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=True)


def _maxdiff(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def test_causal_square_compiles() -> None:
    m = _Causal().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    eager = m(q, k, v)
    cm = lucid.compile(m)
    out = cm(q, k, v)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_causal_non_square_decode_shape_compiles() -> None:
    # Lq < Lk (cached-decode-shaped): bottom-right alignment must match eager.
    m = _Causal().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(1, 2, 4, 64, 16)
    eager = m(q, k, v)
    cm = lucid.compile(m)
    out = cm(q, k, v)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_plain_compiles() -> None:
    m = _Plain().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    eager = m(q, k, v)
    cm = lucid.compile(m)
    out = cm(q, k, v)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_additive_mask_falls_back_but_stays_correct() -> None:
    # The mask isn't wired into the trace, so the emitter must NOT compile a
    # mask-dropping executable — it falls back to eager (correct result).
    m = _MaskOnly().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    mask = metal_tensor(2, 4, 16, 16)
    eager = m(q, k, v, mask)
    cm = lucid.compile(m)
    out = cm(q, k, v, mask)
    assert cm.cache_info()["eager_only"], "masked SDPA must fall back to eager"
    assert _maxdiff(eager, out) == 0.0  # eager path on both sides → bit-exact


def test_causal_plus_mask_falls_back_but_stays_correct() -> None:
    # Eager lets the additive mask win over is_causal; the compile path can't
    # honor the (unwired) mask, so it falls back — matching eager exactly.
    m = _CausalPlusMask().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    mask = metal_tensor(2, 4, 16, 16)
    eager = m(q, k, v, mask)
    cm = lucid.compile(m)
    out = cm(q, k, v, mask)
    assert cm.cache_info()["eager_only"]
    assert _maxdiff(eager, out) == 0.0


def test_multihead_attention_causal_compiles() -> None:
    class _MHA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            out, _ = self.mha(x, x, x, is_causal=True, need_weights=False)
            return out

    m = _MHA().to(COMPILE_DEVICE).eval()
    x = metal_tensor(2, 16, 32)
    eager = m(x)
    cm = lucid.compile(m)
    out = cm(x)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5
