"""
lucid
=====
"""

from lucid.tensor import _ArrayLike, Tensor
from lucid import _func


Tensor.__add__ = _func.add
Tensor.__sub__ = _func.sub
Tensor.__mul__ = _func.mul
Tensor.__truediv__ = _func.truediv
Tensor.__pow__ = _func.pow


def tensor(data: _ArrayLike, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad)


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
