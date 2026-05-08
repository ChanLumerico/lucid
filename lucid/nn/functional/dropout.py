"""
nn.functional dropout operations.
All implementations use the C++ engine ops; no numpy.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# SELU affine constants (from the SELU paper)
_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805
_ALPHA_PRIME = -_SELU_ALPHA * _SELU_SCALE  # ≈ -1.7580993408473766


def dropout(
    x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly zero elements with probability p during training."""
    return _wrap(_C_engine.nn.dropout(_unwrap(x), p, training))


def dropout1d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Randomly zero entire 1-D feature maps (channels) of an
    ``(N, C, L)`` input.  Uses the same engine kernel as ``dropout2d`` /
    ``dropout3d`` — channel-wise masking is rank-agnostic."""
    return _wrap(_C_engine.nn.dropoutnd(_unwrap(x), p, training))


def dropout2d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Randomly zero entire channels with probability p during training."""
    return _wrap(_C_engine.nn.dropoutnd(_unwrap(x), p, training))


def dropout3d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Randomly zero entire 3-D feature maps with probability p during training."""
    return _wrap(_C_engine.nn.dropoutnd(_unwrap(x), p, training))


def alpha_dropout(
    x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Alpha dropout preserving self-normalizing SELU properties.

    During training, each element is either kept (probability 1-p) and
    affine-rescaled, or replaced with alpha' = -alpha*scale.  All arithmetic
    uses engine ops so the result stays on GPU when the input is on GPU.
    """
    if not training or p == 0.0:
        return x

    xi = _unwrap(x)

    if p == 1.0:
        return _wrap(_C_engine.zeros(xi.shape, xi.dtype, xi.device))

    keep_prob = 1.0 - p
    a_coeff = (keep_prob + _ALPHA_PRIME**2 * keep_prob * p) ** (-0.5)
    b_coeff = -a_coeff * _ALPHA_PRIME * p

    # Bernoulli mask (1 = keep, 0 = drop) via uniform sampling
    rand_s = _C_engine.rand(xi.shape, xi.dtype, xi.device)
    thresh = _C_engine.full(xi.shape, keep_prob, xi.dtype, xi.device)
    keep = _C_engine.less(rand_s, thresh)  # bool mask: True → keep
    alpha_t = _C_engine.full(xi.shape, _ALPHA_PRIME, xi.dtype, xi.device)
    mixed = _C_engine.where(keep, xi, alpha_t)  # x or alpha' per element

    # Affine: a * mixed + b
    a_t = _C_engine.full(xi.shape, a_coeff, xi.dtype, xi.device)
    b_t = _C_engine.full(xi.shape, b_coeff, xi.dtype, xi.device)
    out = _C_engine.add(_C_engine.mul(a_t, mixed), b_t)
    return _wrap(out)


def feature_alpha_dropout(
    x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Alpha dropout that zeroes entire feature maps (channels).

    For input of shape (N, C, *), generates a (N, C) Bernoulli mask and
    broadcasts it over the spatial dimensions.  All ops use the C++ engine.
    """
    if not training or p == 0.0:
        return x

    xi = _unwrap(x)
    if len(xi.shape) < 2:
        return alpha_dropout(x, p, training, inplace)

    N, C = int(xi.shape[0]), int(xi.shape[1])
    keep_prob = 1.0 - p

    # (N, C) Bernoulli mask
    rand_nc = _C_engine.rand([N, C], xi.dtype, xi.device)
    thresh = _C_engine.full([N, C], keep_prob, xi.dtype, xi.device)
    keep_nc = _C_engine.less(rand_nc, thresh)  # (N, C) bool

    # Reshape to (N, C, 1, 1, ...) and broadcast_to input shape
    spatial_dims = len(xi.shape) - 2
    mask_shape = [N, C] + [1] * spatial_dims
    keep_broad = _C_engine.reshape(keep_nc, mask_shape)
    keep_broad = _C_engine.broadcast_to(keep_broad, list(xi.shape))

    zeros_t = _C_engine.zeros(xi.shape, xi.dtype, xi.device)
    out = _C_engine.where(keep_broad, xi, zeros_t)
    return _wrap(out)
