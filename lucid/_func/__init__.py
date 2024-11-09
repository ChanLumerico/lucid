from lucid._func import bfunc, ufunc
from lucid._tensor import Tensor
from lucid.types import _Scalar


# Binary functions
def minimum(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise minimum operation"""
    return bfunc.minimum(a, b)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise maximum operation."""
    return bfunc.maximum(a, b)


def power(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise power operation, raises self to the power of other."""
    return bfunc.power(a, b)


def dot(a: Tensor, b: Tensor) -> Tensor:
    """Dot product of two tensors."""
    return bfunc.dot(a, b)


def inner(a: Tensor, b: Tensor) -> Tensor:
    """Inner product of two tensors."""
    return bfunc.inner(a, b)


def outer(a: Tensor, b: Tensor) -> Tensor:
    """Outer product of two tensors."""
    return bfunc.outer(a, b)


# Unary functions
def exp(a: Tensor) -> Tensor:
    """Exponential function"""
    return ufunc.exp(a)


def log(a: Tensor) -> Tensor:
    """Natural logarithm"""
    return ufunc.log(a)


def sqrt(a: Tensor) -> Tensor:
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


def arcsin(a: Tensor) -> Tensor:
    """Arcsin function"""
    return ufunc.arcsin(a)


def arccos(a: Tensor) -> Tensor:
    """Arccos function"""
    return ufunc.arccos(a)


def arctan(a: Tensor) -> Tensor:
    """Arctan function"""
    return ufunc.arctan(a)


def clip(a: Tensor, min_value: _Scalar, max_value: _Scalar) -> Tensor:
    """Clips the values of the tensor to a specified range element-wise."""
    return ufunc.clip(a, min_value, max_value)


def abs(a: Tensor) -> Tensor:
    """Element-wise absolute value."""
    return ufunc.abs(a)


def sign(a: Tensor) -> Tensor:
    """Element-wise sign function."""
    return ufunc.sign(a)


def reciprocal(a: Tensor) -> Tensor:
    """Element-wise reciprocal."""
    return ufunc.reciprocal(a)


def square(a: Tensor) -> Tensor:
    """Element-wise square."""
    return ufunc.square(a)


def cube(a: Tensor) -> Tensor:
    """Element-wise cube."""
    return ufunc.cube(a)


def transpose(a: Tensor, axes: list[int] | None = None) -> Tensor:
    """Transpose over axes."""
    return ufunc.transpose(a, axes)


# Tensor-generating functions
from lucid._func.gen import *


Tensor.__add__ = bfunc._add
Tensor.__radd__ = bfunc._radd
Tensor.__sub__ = bfunc._sub
Tensor.__rsub__ = bfunc._rsub
Tensor.__mul__ = bfunc._mul
Tensor.__rmul__ = bfunc._rmul
Tensor.__truediv__ = bfunc._truediv
Tensor.__rtruediv__ = bfunc._rtruediv
Tensor.__matmul__ = bfunc.matmul

Tensor.__eq__ = bfunc._equal
Tensor.__ne__ = bfunc._not_equal
Tensor.__gt__ = bfunc._greater
Tensor.__ge__ = bfunc._greater_or_equal
Tensor.__lt__ = bfunc._less
Tensor.__le__ = bfunc._less_or_equal

Tensor.__pow__ = ufunc._pow
Tensor.__neg__ = ufunc._neg

Tensor.T = ufunc._T
Tensor.dot = bfunc.dot
Tensor.inner = bfunc.inner
Tensor.outer = bfunc.outer
Tensor.matmul = bfunc.matmul
