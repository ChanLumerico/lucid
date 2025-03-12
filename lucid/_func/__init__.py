from typing import overload, Callable

import lucid
from lucid.types import _Scalar, _ShapeLike, _ArrayLike, _base_dtype

from lucid._tensor import Tensor
from lucid._backend.metal import _is_cpu_op

from lucid._func import bfunc, gfunc, ufunc

from . import metal


# fmt: off
__all__ = [
    "add", "sub", "div", "minimum", "maximum", "power", "dot", "inner", "outer",
    "matmul",
]
# fmt: on


def add(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._add(a, b) if _is_cpu_op(a, b) else metal.bfunc._add(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._sub(a, b) if _is_cpu_op(a, b) else metal.bfunc._sub(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._mul(a, b) if _is_cpu_op(a, b) else metal.bfunc._mul(a, b)


def div(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._truediv(a, b) if _is_cpu_op(a, b) else metal.bfunc._truediv(a, b)


def _equal(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._equal(a, b) if _is_cpu_op(a, b) else metal.bfunc._equal(a, b)


def _not_equal(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._not_equal(a, b) if _is_cpu_op(a, b) else metal.bfunc._not_equal(a, b)


def _greater(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._greater(a, b) if _is_cpu_op(a, b) else metal.bfunc._greater(a, b)


def _greater_or_equal(a: Tensor, b: Tensor) -> Tensor:
    return (
        bfunc._greater_or_equal(a, b)
        if _is_cpu_op(a, b)
        else metal.bfunc._greater_or_equal(a, b)
    )


def _less(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._less(a, b) if _is_cpu_op(a, b) else metal.bfunc._less(a, b)


def _less_or_equal(a: Tensor, b: Tensor) -> Tensor:
    return (
        bfunc._less_or_equal(a, b)
        if _is_cpu_op(a, b)
        else metal.bfunc._less_or_equal(a, b)
    )


def minimum(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.minimum(a, b) if _is_cpu_op(a, b) else metal.bfunc.minimum(a, b)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.maximum(a, b) if _is_cpu_op(a, b) else metal.bfunc.maximum(a, b)


def power(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.power(a, b) if _is_cpu_op(a, b) else metal.bfunc.power(a, b)


def dot(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.dot(a, b) if _is_cpu_op(a, b) else metal.bfunc.dot(a, b)


def inner(a: Tensor, b: Tensor) -> Tensor:
    return bfunc.inner(a, b) if _is_cpu_op(a, b) else metal.bfunc.inner(a, b)


def outer(a: Tensor, b: Tensor) -> Tensor:
    a, b = a.ravel(), b.ravel()
    return bfunc.outer(a, b) if _is_cpu_op(a, b) else metal.bfunc.outer(a, b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return bfunc._matmul(a, b) if _is_cpu_op(a, b) else metal.bfunc._matmul(a, b)


_radd: Callable = lambda self, other: add(self, other)
_rsub: Callable = lambda self, other: sub(other, self)
_rmul: Callable = lambda self, other: mul(self, other)
_rtruediv: Callable = lambda self, other: div(other, self)


def exp(a: Tensor) -> Tensor:
    return ufunc.exp(a)


def log(a: Tensor) -> Tensor:
    return ufunc.log(a)


def log2(a: Tensor) -> Tensor:
    return ufunc.log2(a)


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


def clip(a: Tensor, min_value: _Scalar | None, max_value: _Scalar | None) -> Tensor:
    if min_value is None:
        min_value = lucid.min(a).item()
    if max_value is None:
        max_value = lucid.max(a).item()

    return ufunc.clip(a, min_value, max_value)


def abs(a: Tensor) -> Tensor:
    return ufunc.abs(a)


def sign(a: Tensor) -> Tensor:
    return ufunc.sign(a)


def reciprocal(a: Tensor) -> Tensor:
    return ufunc.reciprocal(a)


def square(a: Tensor) -> Tensor:
    return ufunc.square(a)


def cube(a: Tensor) -> Tensor:
    return ufunc.cube(a)


def transpose(a: Tensor, axes: list[int] | None = None) -> Tensor:
    return ufunc.transpose(a, axes)


def sum(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc.sum(a, axis, keepdims)


def trace(a: Tensor) -> Tensor:
    return ufunc.trace(a)


def mean(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc.mean(a, axis, keepdims)


def var(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc.var(a, axis, keepdims)


def min(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc._min_or_max(a, "min", axis, keepdims)


def max(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc._min_or_max(a, "max", axis, keepdims)


def swapaxes(a: Tensor, axis1: int, axis2: int) -> Tensor:
    return ufunc.swapaxes(a, axis1, axis2)


@overload
def zeros(
    *shape: int,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor: ...


@overload
def zeros(
    shape: _ShapeLike,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor: ...


def zeros(
    *args: int | _ShapeLike,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    shape = lucid._get_overloaded_shape(args)
    return gfunc.zeros(shape, dtype, requires_grad, keep_grad)


def zeros_like(
    a: Tensor | _ArrayLike,
    dtype: type = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return gfunc.zeros_like(a, dtype, requires_grad, keep_grad)


@overload
def ones(
    *shape: int,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor: ...


@overload
def ones(
    shape: _ShapeLike,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor: ...


def ones(
    *args: int | _ShapeLike,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    shape = lucid._get_overloaded_shape(args)
    return gfunc.ones(shape, dtype, requires_grad, keep_grad)


def ones_like(
    a: Tensor | _ArrayLike,
    dtype: type = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return gfunc.ones_like(a, dtype, requires_grad, keep_grad)


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return gfunc.eye(N, M, k, dtype, requires_grad, keep_grad)


def diag(
    v: Tensor | _ArrayLike,
    k: int = 0,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return gfunc.diag(v, k, dtype, requires_grad, keep_grad)


@overload
def arange(
    stop: _Scalar,
    *,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor: ...


@overload
def arange(
    start: _Scalar,
    stop: _Scalar,
    *,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor: ...


@overload
def arange(
    start: _Scalar,
    stop: _Scalar,
    step: _Scalar,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor: ...


def arange(
    *args,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    if len(args) == 1:
        arange_args = (0.0, *args, 1.0)
    elif len(args) == 2:
        arange_args = (*args, 1.0)
    elif len(args) == 3:
        arange_args = (*args,)
    else:
        raise ValueError(f"Expected <=3 arguments got {len(args)} arguments.")

    return gfunc.arange(*arange_args, dtype, requires_grad, keep_grad)


@overload
def empty(
    *shape: int,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor: ...


@overload
def empty(
    shape: _ShapeLike,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor: ...


def empty(
    *args: int | _ShapeLike,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    shape = lucid._get_overloaded_shape(args)
    return gfunc.empty(shape, dtype, requires_grad, keep_grad)


def empty_like(
    a: Tensor | _ArrayLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return gfunc.empty_like(a, dtype, requires_grad, keep_grad)


def linspace(
    start: _Scalar,
    stop: _Scalar,
    num: int = 50,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return gfunc.linspace(start, stop, num, dtype, requires_grad, keep_grad)


Tensor.__add__ = add
Tensor.__radd__ = _radd
Tensor.__sub__ = sub
Tensor.__rsub__ = _rsub
Tensor.__mul__ = mul
Tensor.__rmul__ = _rmul
Tensor.__truediv__ = div
Tensor.__rtruediv__ = _rtruediv
Tensor.__matmul__ = matmul

Tensor.__eq__ = _equal
Tensor.__ne__ = _not_equal
Tensor.__gt__ = _greater
Tensor.__ge__ = _greater_or_equal
Tensor.__lt__ = _less
Tensor.__le__ = _less_or_equal

Tensor.__pow__ = ufunc._pow
Tensor.__neg__ = ufunc._neg

Tensor.T = ufunc._T
Tensor.mT = ufunc._mT
Tensor.transpose = ufunc.transpose
Tensor.dot = bfunc.dot
Tensor.matmul = bfunc._matmul
Tensor.sum = ufunc.sum
Tensor.mean = ufunc.mean
Tensor.var = ufunc.var
Tensor.clip = ufunc.clip
Tensor.swapaxes = ufunc.swapaxes
