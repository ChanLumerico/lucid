"""
nn.functional activation functions.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def relu(x: "Tensor", inplace: bool = False) -> "Tensor":
    """Rectified linear unit activation."""
    return _wrap(_C_engine.relu(_unwrap(x)))


def leaky_relu(
    x: "Tensor", negative_slope: float = 0.01, inplace: bool = False
) -> "Tensor":
    """Leaky rectified linear unit."""
    return _wrap(_C_engine.leaky_relu(_unwrap(x), negative_slope))


def elu(x: "Tensor", alpha: float = 1.0, inplace: bool = False) -> "Tensor":
    """Exponential linear unit."""
    return _wrap(_C_engine.elu(_unwrap(x), alpha))


def selu(x: "Tensor", inplace: bool = False) -> "Tensor":
    """Scaled exponential linear unit."""
    return _wrap(_C_engine.selu(_unwrap(x)))


def gelu(x: "Tensor", approximate: str = "none") -> "Tensor":
    """Gaussian error linear unit."""
    return _wrap(_C_engine.gelu(_unwrap(x)))


def silu(x: "Tensor", inplace: bool = False) -> "Tensor":
    """Sigmoid linear unit (Swish)."""
    return _wrap(_C_engine.silu(_unwrap(x)))


def mish(x: "Tensor") -> "Tensor":
    """Mish activation."""
    return _wrap(_C_engine.mish(_unwrap(x)))


def hardswish(x: "Tensor") -> "Tensor":
    """Hard Swish activation."""
    return _wrap(_C_engine.hard_swish(_unwrap(x)))


def hardsigmoid(x: "Tensor") -> "Tensor":
    """Hard sigmoid activation."""
    return _wrap(_C_engine.hard_sigmoid(_unwrap(x)))


def sigmoid(x: "Tensor") -> "Tensor":
    """Sigmoid activation."""
    return _wrap(_C_engine.sigmoid(_unwrap(x)))


def tanh(x: "Tensor") -> "Tensor":
    """Hyperbolic tangent."""
    return _wrap(_C_engine.tanh(_unwrap(x)))


def softmax(x: "Tensor", dim: int | None = None) -> "Tensor":
    """Softmax along dim."""
    axis = dim if dim is not None else -1
    return _wrap(_C_engine.softmax(_unwrap(x), axis))


def log_softmax(x: "Tensor", dim: int | None = None) -> "Tensor":
    """Log-softmax along dim (numerically stable)."""
    axis = dim if dim is not None else -1
    sm = _C_engine.softmax(_unwrap(x), axis)
    return _wrap(_C_engine.log(sm))


def softplus(x: "Tensor", beta: float = 1.0, threshold: float = 20.0) -> "Tensor":
    """Softplus activation."""
    return _wrap(_C_engine.softplus(_unwrap(x)))


def relu6(x: "Tensor", inplace: bool = False) -> "Tensor":
    """ReLU6 activation (clamped at 6)."""
    return _wrap(_C_engine.relu6(_unwrap(x)))
