from typing import Any
import numpy as np

from lucid._func import bfunc, gfunc, ufunc
from lucid._tensor import Tensor
from lucid.types import _Scalar, _ShapeLike, _ArrayLike


def add(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._add(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._sub(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._mul(a, b)


def div(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._truediv(a, b)


def minimum(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.minimum(a, b)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.maximum(a, b)


def power(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.power(a, b)


def dot(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.dot(a, b)


def inner(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.inner(a, b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.matmul(a, b)


def exp(a: Tensor) -> Tensor:
    return ufunc.exp(a)


def log(a: Tensor) -> Tensor:
    return ufunc.log(a)


def sqrt(a: Tensor) -> Tensor:
    return ufunc.sqrt(a)


def sin(a: Tensor) -> Tensor:
    return ufunc.sin(a)


def cos(a: Tensor) -> Tensor:
    return ufunc.cos(a)


def tan(a: Tensor) -> Tensor:
    return ufunc.tan(a)


def arcsin(a: Tensor) -> Tensor:
    return ufunc.arcsin(a)


def arccos(a: Tensor) -> Tensor:
    return ufunc.arccos(a)


def arctan(a: Tensor) -> Tensor:
    return ufunc.arctan(a)


def sinh(a: Tensor) -> Tensor:
    return ufunc.sinh(a)


def cosh(a: Tensor) -> Tensor:
    return ufunc.cosh(a)


def tanh(a: Tensor) -> Tensor:
    return ufunc.tanh(a)


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


def sum(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    """Sum along the specified axis."""
    return ufunc.sum(a, axis, keepdims)


def trace(a: Tensor) -> Tensor:
    """Trace operation."""
    return ufunc.trace(a)


def mean(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc.mean(a, axis, keepdims)


def var(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    """Compute the variance along the specified axis; if `axis=None`,
    variance of the entire tensor is returned."""
    return ufunc.var(a, axis, keepdims)


def zeros(
    shape: _ShapeLike,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create a zero-tensor with the specified shape."""
    return gfunc.zeros(shape, dtype, requires_grad, keep_grad)


def zeros_like(
    a: Tensor | _ArrayLike,
    dtype: Any = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create a zero-tensor of shape same with the given tensor."""
    return zeros_like(a, dtype, requires_grad, keep_grad)


def ones(
    shape: _ShapeLike,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create an one-tensor with the specified shape."""
    return gfunc.ones(shape, dtype, requires_grad, keep_grad)


def ones_like(
    a: Tensor | _ArrayLike,
    dtype: Any = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create an one-tensor of shape same with the given tensor."""
    return gfunc.ones_like(a, dtype, requires_grad, keep_grad)


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create an identical matrix of shape `(N, M)`."""
    return gfunc.eye(N, M, k, dtype, requires_grad, keep_grad)


def diag(
    v: Tensor | _ArrayLike,
    k: int = 0,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create a diagonal matrix from the given vector."""
    return gfunc.diag(v, k, dtype, requires_grad, keep_grad)


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
Tensor.sum = ufunc.sum
Tensor.mean = ufunc.mean
Tensor.var = ufunc.var
