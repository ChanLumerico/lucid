"""
Operations for `Tensor`
"""

from lucid import _func
from lucid.tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition."""
    return _func.add(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction."""
    return _func.sub(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication."""
    return _func.mul(a, b)


def div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division."""
    return _func.truediv(a, b)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise maximum operation."""
    return _func.maximum(a, b)


def pow(a: Tensor, exp: int | float) -> Tensor:
    """Element-wise power operation."""
    return _func.pow(a, exp)
