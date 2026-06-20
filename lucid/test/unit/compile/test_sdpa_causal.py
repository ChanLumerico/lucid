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
  2. **Additive float masks compile** — the tracer wires the auxiliary mask
     tensor into the graph (a non-differentiable SDPA input), so masked
     attention (BERT padding masks, cross-attention) compiles and adds the
     mask before the softmax.  A bool mask, which the eager backend treats as
     a set-mask rather than an additive one, still falls back to eager.
"""

import lucid
import lucid.models as M
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
    def forward(
        self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor
    ) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class _Plain(nn.Module):
    def forward(
        self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor
    ) -> lucid.Tensor:
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


def test_additive_float_mask_compiles() -> None:
    # An additive float mask is wired into the trace (a non-differentiable
    # auxiliary SDPA input) and added before the softmax, so masked attention
    # — e.g. BERT padding masks — compiles instead of falling back to eager.
    m = _MaskOnly().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    mask = metal_tensor(2, 4, 16, 16)
    eager = m(q, k, v, mask)
    cm = lucid.compile(m)
    out = cm(q, k, v, mask)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_causal_plus_mask_lets_mask_win() -> None:
    # Eager lets the additive mask win over is_causal (its fused-causal path is
    # bypassed once a float attn_mask is supplied).  The emitter mirrors that:
    # the mask is added and the causal block is skipped, so it compiles and
    # matches eager.
    m = _CausalPlusMask().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 16, 16, 16)
    mask = metal_tensor(2, 4, 16, 16)
    eager = m(q, k, v, mask)
    cm = lucid.compile(m)
    out = cm(q, k, v, mask)
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5


def test_bool_mask_falls_back_to_eager() -> None:
    # A bool mask is a set-mask in eager (−inf where false), not an additive
    # add — the emitter can't reproduce that with a plain add, so it bails to
    # eager rather than emit a wrong executable.
    class _BoolMask(nn.Module):
        def forward(
            self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor, m: lucid.Tensor
        ) -> lucid.Tensor:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

    m = _BoolMask().to(COMPILE_DEVICE).eval()
    q, k, v = _qkv(2, 4, 8, 8, 16)
    bool_mask = lucid.tril(lucid.ones(8, 8, device=COMPILE_DEVICE)).to(lucid.bool_)
    eager = m(q, k, v, bool_mask)
    cm = lucid.compile(m)
    out = cm(q, k, v, bool_mask)
    assert cm.cache_info()["eager_only"], "bool-mask SDPA must fall back to eager"
    assert _maxdiff(eager, out) == 0.0


def test_multihead_attention_causal_compiles() -> None:
    class _MHA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(
                embed_dim=32, num_heads=4, batch_first=True
            )

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


def test_bert_with_padding_mask_compiles() -> None:
    # The flagship payoff: BERT calls F.scaled_dot_product_attention with an
    # additive padding mask, so wiring the mask into the trace lets a real
    # masked transformer compile into one executable (was eager-fallback).
    cfg = M.BERTConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    model = M.BERTModel(cfg).to(COMPILE_DEVICE).eval()

    b, t = 2, 8
    # token ids in [0, b*t) < vocab_size (64); arange is float, embedding
    # needs integer indices.
    ids = lucid.arange(0, b * t, device=COMPILE_DEVICE).reshape(b, t).to(lucid.int64)
    mask = lucid.ones(b, t, device=COMPILE_DEVICE)
    mask[0, 6:] = 0.0  # pad the tail of the first sequence

    eager = model(ids, attention_mask=mask).last_hidden_state
    cm = lucid.compile(model)
    out = cm(ids, attention_mask=mask).last_hidden_state
    info = cm.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert _maxdiff(eager, out) < 1e-5
