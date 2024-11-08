from lucid._func import bfunc, ufunc
from lucid.tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition."""
    return bfunc.add(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction."""
    return bfunc.sub(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication."""
    return bfunc.mul(a, b)


def div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division."""
    return bfunc.truediv(a, b)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise maximum operation."""
    return bfunc.maximum(a, b)


def pow(a: Tensor, exp: int | float) -> Tensor:
    """Element-wise power operation."""
    return ufunc.pow(a, exp)


Tensor.__add__ = bfunc.add
Tensor.__radd__ = bfunc.radd
Tensor.__sub__ = bfunc.sub
Tensor.__rsub__ = bfunc.rsub
Tensor.__mul__ = bfunc.mul
Tensor.__rmul__ = bfunc.rmul
Tensor.__truediv__ = bfunc.truediv
Tensor.__rtrudiv__ = bfunc.rtruediv
Tensor.__pow__ = ufunc.pow
