"""
nn.functional linear (fully-connected) operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def linear(
    x: "Tensor",
    weight: "Tensor",
    bias: "Tensor | None" = None,
) -> "Tensor":
    """
    Apply a linear transformation: y = x @ weight.T + bias.

    Args:
        x:      (..., in_features)
        weight: (out_features, in_features)
        bias:   (out_features,) or None
    """
    w_impl = _unwrap(weight)
    out_features = w_impl.shape[0]
    if bias is not None:
        b_impl = _unwrap(bias)
    else:
        b_impl = _C_engine.zeros([out_features], w_impl.dtype, w_impl.device)
    return _wrap(_C_engine.nn.linear(_unwrap(x), w_impl, b_impl))


def bilinear(
    x1: "Tensor",
    x2: "Tensor",
    weight: "Tensor",
    bias: "Tensor | None" = None,
) -> "Tensor":
    """Bilinear transformation: y = x1 @ W @ x2.T + bias."""
    result = _wrap(_C_engine.nn.bilinear_layer(_unwrap(x1), _unwrap(x2), _unwrap(weight)))
    if bias is not None:
        result = _wrap(_C_engine.add(result._impl, _unwrap(bias)))
    return result
