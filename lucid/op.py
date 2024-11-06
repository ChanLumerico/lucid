"""
Operations for `Tensor`
"""

from lucid import _func
from lucid.tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    return _func.add(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    return _func.sub(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return _func.mul(a, b)


def div(a: Tensor, b: Tensor) -> Tensor:
    return _func.truediv(a, b)


def pow(a: Tensor, exp: float) -> Tensor:
    return _func.pow(a, exp)
