"""
lucid.ops.bfunc — binary operations (mirrors `lucid_legacy/_func/bfunc.py`).

All forwards are thin wrappers around `_C_engine.{op}(a._impl, b._impl)`,
re-wrapped as `Tensor`. Backward / autograd graph wiring is done entirely
inside the C++ engine: the returned TensorImpl's `grad_fn_` is set
automatically when any input has `requires_grad`.

Includes:
  * Element-wise arithmetic: add, sub, multiply, div (with floor=), power,
    minimum, maximum
  * Linear algebra contraction: matmul, dot, inner, outer, tensordot
  * Comparisons (return Bool): _equal, _not_equal, _greater,
    _greater_or_equal, _less, _less_or_equal
  * Bitwise: _bitwise_and, _bitwise_or
  * Floor division: floordiv (also reachable via `div(..., floor=True)`)
  * In-place variants: add_, sub_, mul_, div_, power_, minimum_, maximum_
"""

from __future__ import annotations

from typing import Callable

from lucid._C import engine as _C_engine
from lucid._tensor import Tensor
from lucid._bridge import impl_of


__all__ = [
    # Arithmetic
    "add", "sub", "multiply", "div", "minimum", "maximum", "power",
    # In-place arithmetic
    "add_", "sub_", "mul_", "div_", "minimum_", "maximum_", "power_",
    # Linear algebra contraction
    "matmul", "dot", "inner", "outer", "tensordot",
    # Comparisons
    "_equal", "_not_equal", "_greater", "_greater_or_equal",
    "_less", "_less_or_equal",
    # Bitwise
    "_bitwise_and", "_bitwise_or",
    # Reverse-protocol helpers (re-bound on Tensor in __init__)
    "_radd", "_rsub", "_rmul", "_rtruediv",
    "_floordiv", "_rfloordiv",
    "_rbitwise_and", "_rbitwise_or",
]


# --------------------------------------------------------------------------- #
# Element-wise arithmetic
# --------------------------------------------------------------------------- #

def add(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.add(impl_of(a), impl_of(b)))


def sub(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.sub(impl_of(a), impl_of(b)))


def multiply(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.mul(impl_of(a), impl_of(b)))


def div(a: Tensor, b: Tensor, /, floor: bool = False) -> Tensor:
    if floor:
        return Tensor._wrap(_C_engine.floordiv(impl_of(a), impl_of(b)))
    return Tensor._wrap(_C_engine.div(impl_of(a), impl_of(b)))


def minimum(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.minimum(impl_of(a), impl_of(b)))


def maximum(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.maximum(impl_of(a), impl_of(b)))


def power(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.pow(impl_of(a), impl_of(b)))


# --------------------------------------------------------------------------- #
# In-place variants — mutate `a`'s storage and bump version
# --------------------------------------------------------------------------- #

def add_(a: Tensor, b: Tensor, /) -> Tensor:
    _C_engine.add_(impl_of(a), impl_of(b))
    return a


def sub_(a: Tensor, b: Tensor, /) -> Tensor:
    _C_engine.sub_(impl_of(a), impl_of(b))
    return a


def mul_(a: Tensor, b: Tensor, /) -> Tensor:
    _C_engine.mul_(impl_of(a), impl_of(b))
    return a


def div_(a: Tensor, b: Tensor, /, floor: bool = False) -> Tensor:
    if floor:
        a._impl = _C_engine.floordiv(impl_of(a), impl_of(b))
        return a
    _C_engine.div_(impl_of(a), impl_of(b))
    return a


def minimum_(a: Tensor, b: Tensor, /) -> Tensor:
    _C_engine.minimum_(impl_of(a), impl_of(b))
    return a


def maximum_(a: Tensor, b: Tensor, /) -> Tensor:
    _C_engine.maximum_(impl_of(a), impl_of(b))
    return a


def power_(a: Tensor, b: Tensor, /) -> Tensor:
    _C_engine.pow_(impl_of(a), impl_of(b))
    return a


# --------------------------------------------------------------------------- #
# Linear algebra contractions
# --------------------------------------------------------------------------- #

def matmul(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.matmul(impl_of(a), impl_of(b)))


def dot(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.dot(impl_of(a), impl_of(b)))


def inner(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.inner(impl_of(a), impl_of(b)))


def outer(a: Tensor, b: Tensor, /) -> Tensor:
    # numpy.outer flattens both inputs first.
    a = a.ravel()
    b = b.ravel()
    return Tensor._wrap(_C_engine.outer(impl_of(a), impl_of(b)))


def tensordot(
    a: Tensor,
    b: Tensor,
    /,
    axes: int | tuple[int, int] | tuple[list[int], list[int]] = 2,
) -> Tensor:
    if isinstance(axes, int):
        n = axes
        axes_a = list(range(a.ndim - n, a.ndim))
        axes_b = list(range(n))
    elif isinstance(axes, tuple) and len(axes) == 2:
        x, y = axes
        if isinstance(x, int) and isinstance(y, int):
            axes_a, axes_b = [x], [y]
        else:
            axes_a, axes_b = list(x), list(y)
    else:
        raise TypeError(f"tensordot: bad axes spec {axes!r}")
    return Tensor._wrap(
        _C_engine.tensordot(impl_of(a), impl_of(b), axes_a, axes_b)
    )


# --------------------------------------------------------------------------- #
# Comparisons (return bool tensor)
# --------------------------------------------------------------------------- #

def _equal(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.equal(impl_of(a), impl_of(b)))


def _not_equal(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.not_equal(impl_of(a), impl_of(b)))


def _greater(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.greater(impl_of(a), impl_of(b)))


def _greater_or_equal(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.greater_equal(impl_of(a), impl_of(b)))


def _less(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.less(impl_of(a), impl_of(b)))


def _less_or_equal(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.less_equal(impl_of(a), impl_of(b)))


# --------------------------------------------------------------------------- #
# Bitwise (integer / bool only)
# --------------------------------------------------------------------------- #

def _bitwise_and(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.bitwise_and(impl_of(a), impl_of(b)))


def _bitwise_or(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.bitwise_or(impl_of(a), impl_of(b)))


# --------------------------------------------------------------------------- #
# Reverse-protocol helpers (used by Tensor's __radd__, __rsub__, ...)
# --------------------------------------------------------------------------- #

_radd:        Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: add(a, b)
_rsub:        Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: sub(b, a)
_rmul:        Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: multiply(a, b)
_rtruediv:    Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: div(b, a)
_floordiv:    Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: div(a, b, floor=True)
_rfloordiv:   Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: div(b, a, floor=True)
_rbitwise_and: Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: _bitwise_and(b, a)
_rbitwise_or:  Callable[[Tensor, Tensor], Tensor] = lambda a, b, /: _bitwise_or(b, a)
