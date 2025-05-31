from typing import overload, Callable

import lucid
from lucid.types import (
    _Scalar,
    _ShapeLike,
    _ArrayLike,
    _DeviceType,
    _BuiltinNumeric,
    Numeric,
)

from lucid._tensor import Tensor
from lucid._func import bfunc, gfunc, ufunc


# fmt: off
__all__ = [
    "add", "sub", "multiply", "div", "minimum", "maximum", "power", "dot", "inner", 
    "outer", "matmul", 
    
    "exp", "log", "log2", "sqrt", "sin", "cos", "tan", "arcsin", "arccos", "arctan", 
    "sinh", "cosh", "tanh", "clip", "abs", "sign", "reciprocal", "square", "cube",
    "transpose", "sum", "trace", "mean", "var", "min", "max", "swapaxes", "round",
    "floor", "ceil",

    "zeros", "zeros_like", "ones", "ones_like", "eye", "diag", "arange", "empty",
    "empty_like", "linspace", "full", "full_like",
]
# fmt: on


def add(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.add()(a, b)


def sub(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.sub()(a, b)


def multiply(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.multiply()(a, b)


def div(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.truediv()(a, b)


def _equal(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc._equal()(a, b)


def _not_equal(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc._not_equal()(a, b)


def _greater(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc._greater()(a, b)


def _greater_or_equal(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc._greater_or_equal()(a, b)


def _less(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc._less()(a, b)


def _less_or_equal(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc._less_or_equal()(a, b)


def minimum(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.minimum()(a, b)


def maximum(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.maximum()(a, b)


def power(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.power()(a, b)


def dot(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.dot()(a, b)


def inner(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.inner()(a, b)


def outer(a: Tensor, b: Tensor, /) -> Tensor:
    a, b = a.ravel(), b.ravel()
    return bfunc.outer()(a, b)


def matmul(a: Tensor, b: Tensor, /) -> Tensor:
    return bfunc.matmul()(a, b)


def __check_int_bool_dtype(*ts: Tensor) -> None:
    if not all(t.dtype is bool or t.dtype.base_dtype is int for t in ts):
        raise TypeError(
            f"All tensors must be int or boolean type for bitwise operations."
        )


def _bitwise_and(a: Tensor, b: Tensor, /) -> Tensor:
    __check_int_bool_dtype(a, b)
    return bfunc._bitwise_and()(a, b)


def _bitwise_or(a: Tensor, b: Tensor, /) -> Tensor:
    __check_int_bool_dtype(a, b)
    return bfunc._bitwise_or()(a, b)


_radd: Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: add(a, b)
_rsub: Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: sub(b, a)
_rmul: Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: multiply(a, b)
_rtruediv: Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: div(b, a)
_rbitwise_and: Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: _bitwise_and(b, a)
_rbitwise_or: Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: _bitwise_or(b, a)


def _pow(a: Tensor, /, exp: _Scalar) -> Tensor:
    return ufunc._pow(exp)(a)


def _neg(a: Tensor, /) -> Tensor:
    return ufunc._neg()(a)


def exp(a: Tensor, /) -> Tensor:
    return ufunc.exp()(a)


def log(a: Tensor, /) -> Tensor:
    return ufunc.log()(a)


def log2(a: Tensor, /) -> Tensor:
    return ufunc.log2()(a)


def sqrt(a: Tensor, /) -> Tensor:
    return ufunc.sqrt()(a)


def sin(a: Tensor, /) -> Tensor:
    return ufunc.sin()(a)


def cos(a: Tensor, /) -> Tensor:
    return ufunc.cos()(a)


def tan(a: Tensor, /) -> Tensor:
    return ufunc.tan()(a)


def arcsin(a: Tensor, /) -> Tensor:
    return ufunc.arcsin()(a)


def arccos(a: Tensor, /) -> Tensor:
    return ufunc.arccos()(a)


def arctan(a: Tensor, /) -> Tensor:
    return ufunc.arctan()(a)


def sinh(a: Tensor, /) -> Tensor:
    return ufunc.sinh()(a)


def cosh(a: Tensor, /) -> Tensor:
    return ufunc.cosh()(a)


def tanh(a: Tensor, /) -> Tensor:
    return ufunc.tanh()(a)


def clip(a: Tensor, /, min_value: _Scalar | None, max_value: _Scalar | None) -> Tensor:
    if min_value is None:
        min_value = lucid.min(a).item()
    if max_value is None:
        max_value = lucid.max(a).item()

    return ufunc.clip(min_value, max_value)(a)


def abs(a: Tensor, /) -> Tensor:
    return ufunc._abs()(a)


def sign(a: Tensor, /) -> Tensor:
    return ufunc.sign()(a)


def reciprocal(a: Tensor, /) -> Tensor:
    return ufunc.reciprocal()(a)


def square(a: Tensor, /) -> Tensor:
    return ufunc.square()(a)


def cube(a: Tensor, /) -> Tensor:
    return ufunc.cube()(a)


@property
def _T(a: Tensor, /) -> Tensor:
    return ufunc._T()(a)


@property
def _mT(a: Tensor, /) -> Tensor:
    return ufunc._mT()(a)


def transpose(a: Tensor, /, axes: list[int] | None = None) -> Tensor:
    return ufunc.transpose(axes, a.ndim)(a)


def sum(
    a: Tensor, /, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc.sum(axis, keepdims)(a)


def trace(a: Tensor, /) -> Tensor:
    return ufunc.trace()(a)


def mean(
    a: Tensor, /, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc.mean(axis, keepdims)(a)


def var(
    a: Tensor, /, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc.var(axis, keepdims)(a)


def min(
    a: Tensor, /, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc._min_or_max("min", axis, keepdims)(a)


def max(
    a: Tensor, /, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    return ufunc._min_or_max("max", axis, keepdims)(a)


def swapaxes(a: Tensor, /, axis1: int, axis2: int) -> Tensor:
    return ufunc.swapaxes(axis1, axis2)(a)


def round(a: Tensor, /, decimals: int = 0) -> Tensor:
    return ufunc.round(decimals)(a)


def floor(a: Tensor) -> Tensor:
    return ufunc.floor()(a)


def ceil(a: Tensor) -> Tensor:
    return ufunc.ceil()(a)


@overload
def zeros(
    *shape: int,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


@overload
def zeros(
    shape: _ShapeLike,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


def zeros(
    *args: int | _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    shape = lucid._get_overloaded_shape(args)
    return gfunc.zeros(shape, dtype, requires_grad, keep_grad, device)


def zeros_like(
    a: Tensor | _ArrayLike,
    /,
    dtype: type = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    return gfunc.zeros_like(a, dtype, requires_grad, keep_grad, device)


@overload
def ones(
    *shape: int,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


@overload
def ones(
    shape: _ShapeLike,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


def ones(
    *args: int | _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    shape = lucid._get_overloaded_shape(args)
    return gfunc.ones(shape, dtype, requires_grad, keep_grad, device)


def ones_like(
    a: Tensor | _ArrayLike,
    /,
    dtype: type = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    return gfunc.ones_like(a, dtype, requires_grad, keep_grad, device)


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return gfunc.eye(N, M, k, dtype, requires_grad, keep_grad, device)


def diag(
    v: Tensor | _ArrayLike,
    k: int = 0,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    return gfunc.diag(v, k, dtype, requires_grad, keep_grad, device)


@overload
def arange(
    stop: _Scalar,
    *,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


@overload
def arange(
    start: _Scalar,
    stop: _Scalar,
    *,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


@overload
def arange(
    start: _Scalar,
    stop: _Scalar,
    step: _Scalar,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


def arange(
    *args,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    if len(args) == 1:
        arange_args = (0.0, *args, 1.0)
    elif len(args) == 2:
        arange_args = (*args, 1.0)
    elif len(args) == 3:
        arange_args = (*args,)
    else:
        raise ValueError(f"Expected <=3 arguments got {len(args)} arguments.")

    return gfunc.arange(*arange_args, dtype, requires_grad, keep_grad, device)


@overload
def empty(
    *shape: int,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


@overload
def empty(
    shape: _ShapeLike,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor: ...


def empty(
    *args: int | _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    shape = lucid._get_overloaded_shape(args)
    return gfunc.empty(shape, dtype, requires_grad, keep_grad, device)


def empty_like(
    a: Tensor | _ArrayLike,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    return gfunc.empty_like(a, dtype, requires_grad, keep_grad, device)


def linspace(
    start: _Scalar,
    stop: _Scalar,
    num: int = 50,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return gfunc.linspace(start, stop, num, dtype, requires_grad, keep_grad, device)


def full(
    shape: int | _ShapeLike,
    fill_value: _Scalar,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return gfunc.full(shape, fill_value, dtype, requires_grad, keep_grad, device)


def full_like(
    a: Tensor | _ArrayLike,
    fill_value: _Scalar,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    return gfunc.full_like(a, fill_value, dtype, requires_grad, keep_grad, device)


Tensor.__add__ = add
Tensor.__radd__ = _radd
Tensor.__sub__ = sub
Tensor.__rsub__ = _rsub
Tensor.__mul__ = multiply
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

Tensor.__pow__ = _pow
Tensor.__neg__ = _neg

Tensor.__and__ = _bitwise_and
Tensor.__rand__ = _rbitwise_and
Tensor.__or__ = _bitwise_or
Tensor.__ror__ = _rbitwise_or

Tensor.T = _T
Tensor.mT = _mT
Tensor.transpose = transpose
Tensor.dot = dot
Tensor.matmul = matmul
Tensor.sum = sum
Tensor.mean = mean
Tensor.var = var
Tensor.clip = clip
Tensor.swapaxes = swapaxes
Tensor.round = round
Tensor.floor = floor
Tensor.ceil = ceil
