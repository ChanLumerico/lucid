"""
nn.functional attention operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    """
    Scaled dot-product attention.

    Args:
        query:     (B, H, T, E)
        key:       (B, H, S, E)
        value:     (B, H, S, V)
        attn_mask: optional additive mask (B, H, T, S)
        dropout_p: dropout probability applied to attention weights
        is_causal: apply causal (triangular) mask
        scale:     optional softmax scale (default: 1/sqrt(E))
    """
    import math

    mask = _unwrap(attn_mask) if attn_mask is not None else None
    head_dim = query.shape[-1]
    scale_val = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    return _wrap(
        _C_engine.nn.scaled_dot_product_attention(
            _unwrap(query),
            _unwrap(key),
            _unwrap(value),
            mask,
            scale_val,
            is_causal,
        )
    )
