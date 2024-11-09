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


def minimum(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise minimum operation"""
    return bfunc.minimum(a, b)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise maximum operation."""
    return bfunc.maximum(a, b)


def power(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise power operation, raises self to the power of other."""
    return bfunc.power(a, b)


def pow(a: Tensor, exp: int | float) -> Tensor:
    """Element-wise power operation."""
    return ufunc.pow(a, exp)


def exp(a: Tensor) -> Tensor:
    """Exponential function"""
    return ufunc.exp(a)


def log(a: Tensor) -> Tensor:
    """Natural logarithm"""
    return ufunc.log(a)


def sqrr(a: Tensor) -> Tensor:
    """Square root"""
    return ufunc.sqrt(a)


def sin(a: Tensor) -> Tensor:
    """Sine function"""
    return ufunc.sin(a)


def cos(a: Tensor) -> Tensor:
    """Cosine function"""
    return ufunc.cos(a)


def tan(a: Tensor) -> Tensor:
    """Tangent function"""
    return ufunc.tan(a)


def clip(a: Tensor, min_value: float, max_value: float) -> Tensor:
    """Clips the values of the tensor to a specified range element-wise."""
    return ufunc.clip(a, min_value, max_value)


Tensor.__add__ = bfunc.add
Tensor.__radd__ = bfunc.radd
Tensor.__sub__ = bfunc.sub
Tensor.__rsub__ = bfunc.rsub
Tensor.__mul__ = bfunc.mul
Tensor.__rmul__ = bfunc.rmul
Tensor.__truediv__ = bfunc.truediv
Tensor.__rtrudiv__ = bfunc.rtruediv
Tensor.__eq__ = bfunc.equal
Tensor.__gt__ = bfunc.greater
Tensor.__lt__ = bfunc.less

Tensor.__pow__ = ufunc.pow
Tensor.__neg__ = ufunc.neg
