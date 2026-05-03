"""
nn.functional activation functions.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def relu(x: Tensor, inplace: bool = False) -> Tensor:
    """Rectified linear unit activation."""
    return _wrap(_C_engine.relu(_unwrap(x)))


def leaky_relu(
    x: Tensor, negative_slope: float = 0.01, inplace: bool = False
) -> Tensor:
    """Leaky rectified linear unit."""
    return _wrap(_C_engine.leaky_relu(_unwrap(x), negative_slope))


def elu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """Exponential linear unit."""
    return _wrap(_C_engine.elu(_unwrap(x), alpha))


def selu(x: Tensor, inplace: bool = False) -> Tensor:
    """Scaled exponential linear unit."""
    return _wrap(_C_engine.selu(_unwrap(x)))


def gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """Gaussian error linear unit."""
    return _wrap(_C_engine.gelu(_unwrap(x)))


def silu(x: Tensor, inplace: bool = False) -> Tensor:
    """Sigmoid linear unit (Swish)."""
    return _wrap(_C_engine.silu(_unwrap(x)))


def mish(x: Tensor) -> Tensor:
    """Mish activation."""
    return _wrap(_C_engine.mish(_unwrap(x)))


def hardswish(x: Tensor) -> Tensor:
    """Hard Swish activation."""
    return _wrap(_C_engine.hard_swish(_unwrap(x)))


def hardsigmoid(x: Tensor) -> Tensor:
    """Hard sigmoid activation."""
    return _wrap(_C_engine.hard_sigmoid(_unwrap(x)))


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation."""
    return _wrap(_C_engine.sigmoid(_unwrap(x)))


def tanh(x: Tensor) -> Tensor:
    """Hyperbolic tangent."""
    return _wrap(_C_engine.tanh(_unwrap(x)))


def softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Softmax along dim."""
    axis = dim if dim is not None else -1
    return _wrap(_C_engine.softmax(_unwrap(x), axis))


def log_softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Log-softmax along dim (numerically stable)."""
    axis = dim if dim is not None else -1
    sm = _C_engine.softmax(_unwrap(x), axis)
    return _wrap(_C_engine.log(sm))


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Softplus activation."""
    return _wrap(_C_engine.softplus(_unwrap(x)))


def relu6(x: Tensor, inplace: bool = False) -> Tensor:
    """ReLU6 activation (clamped at 6)."""
    return _wrap(_C_engine.relu6(_unwrap(x)))


def softmin(x: Tensor, dim: int | None = None) -> Tensor:
    """Softmin: softmax applied to the negation of x."""
    axis = dim if dim is not None else -1
    return _wrap(_C_engine.softmax(_C_engine.neg(_unwrap(x)), axis))


def glu(x: Tensor, dim: int = -1) -> Tensor:
    """Gated linear unit: splits x along dim, returns first * sigmoid(second)."""
    impl = _unwrap(x)
    n = impl.shape[dim] // 2
    # Split into two halves along dim
    parts = _C_engine.split_at(impl, [n], dim)
    first, second = parts[0], parts[1]
    return _wrap(_C_engine.mul(first, _C_engine.sigmoid(second)))


def prelu(x: Tensor, weight: Tensor) -> Tensor:
    """Parametric ReLU: max(0,x) + weight * min(0,x)."""
    xi = _unwrap(x)
    wi = _unwrap(weight)
    pos = _C_engine.relu(xi)
    neg_part = _C_engine.mul(wi, _C_engine.minimum(_C_engine.zeros(xi.shape, xi.dtype, xi.device), xi))
    return _wrap(_C_engine.add(pos, neg_part))


def normalize(
    x: Tensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
) -> Tensor:
    """L_p normalize x along dim."""
    return _wrap(_C_engine.nn.lp_normalize(_unwrap(x), p, dim, eps))


def cosine_similarity(
    x1: Tensor,
    x2: Tensor,
    dim: int = 1,
    eps: float = 1e-8,
) -> Tensor:
    """Compute cosine similarity along dim."""
    x1n = normalize(x1, p=2.0, dim=dim, eps=eps)
    x2n = normalize(x2, p=2.0, dim=dim, eps=eps)
    # Element-wise product then sum along dim
    impl = _C_engine.sum(_C_engine.mul(_unwrap(x1n), _unwrap(x2n)), [dim])
    return _wrap(impl)


def pairwise_distance(
    x1: Tensor,
    x2: Tensor,
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False,
) -> Tensor:
    """Compute pairwise L_p distance between x1 and x2."""
    diff = _C_engine.sub(_unwrap(x1), _unwrap(x2))
    # |diff|_p = (sum |diff|^p)^(1/p)
    abs_diff = _C_engine.abs(diff)
    powered = _C_engine.pow(_C_engine.add(abs_diff, _C_engine.full(abs_diff.shape, eps, abs_diff.dtype, abs_diff.device)),
                            _C_engine.full(abs_diff.shape, p, abs_diff.dtype, abs_diff.device))
    s = _C_engine.sum(powered, -1)
    inv_p = _C_engine.full(s.shape, 1.0 / p, s.dtype, s.device)
    dist = _C_engine.pow(s, inv_p)
    if keepdim:
        dist = _C_engine.unsqueeze(dist, -1)
    return _wrap(dist)
