"""
lucid.nn.functional._attention — fused scaled-dot-product attention.

The fast path is the C++ kernel `_C_nn.scaled_dot_product_attention` (one
fused forward + backward, ~2 GEMMs + softmax). When the caller asks for the
attention weights or applies dropout on them, we fall back to a composition
of C++ ops (matmul / softmax / masked_fill / dropout) — still 100% C++.
"""

from __future__ import annotations

import math

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of


def _resolve_scale(query: Tensor, scale: float | None) -> float:
    if scale is not None:
        return float(scale)
    return 1.0 / math.sqrt(query.shape[-1])


def _composed_path(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float,
    return_weights: bool,
) -> Tensor | tuple[Tensor, Tensor]:
    # Pure-C++ composition. Used when dropout_p > 0 (attention weight
    # dropout has its own RNG path) or when caller wants the weights.
    import lucid

    scores = (query @ key.mT) * scale

    if is_causal:
        L_q, L_k = scores.shape[-2], scores.shape[-1]
        causal = lucid.triu(
            lucid.ones((L_q, L_k), dtype=scores.dtype, device=scores.device),
            diagonal=1,
        )
        scores = lucid.masked_fill(scores, causal.astype(lucid.Bool), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype is lucid.Bool:
            scores = lucid.masked_fill(scores, attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    weights = lucid.nn.functional.softmax(scores, axis=-1)
    if dropout_p > 0.0:
        weights = lucid.nn.functional.dropout(weights, dropout_p, training=True)

    output = weights @ value
    if return_weights:
        return output, weights
    return output


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    output_weight: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    s = _resolve_scale(query, scale)

    if dropout_p > 0.0:
        return _composed_path(query, key, value, attn_mask, dropout_p,
                              is_causal, s, output_weight)

    if output_weight:
        out_impl, w_impl = _C_nn.scaled_dot_product_attention_with_weights(
            impl_of(query), impl_of(key), impl_of(value),
            impl_of(attn_mask) if attn_mask is not None else None,
            s, is_causal,
        )
        return Tensor._wrap(out_impl), Tensor._wrap(w_impl)

    return Tensor._wrap(_C_nn.scaled_dot_product_attention(
        impl_of(query), impl_of(key), impl_of(value),
        impl_of(attn_mask) if attn_mask is not None else None,
        s, is_causal,
    ))
