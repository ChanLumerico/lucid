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
    # numpy.inner: contract over the last axis of each operand.  Routed
    # through einsum so the autograd chain is intact (engine.inner has no
    # backward).
    if a.ndim == 0 or b.ndim == 0:
        return multiply(a, b)
    if a.shape[-1] != b.shape[-1]:
        from lucid.error import ShapeMismatch  # fallback to engine error type
        raise ValueError(
            f"inner: last dims must match, got {a.shape} vs {b.shape}")
    if a.ndim + b.ndim > 26:
        raise NotImplementedError("inner: too many axes for label set")
    a_labels = list("abcdefghijklmnopqrstuvwxyz"[:a.ndim])
    b_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXY"[:b.ndim - 1]) + [a_labels[-1]]
    out_labels = a_labels[:-1] + b_labels[:-1]
    pattern = f"{''.join(a_labels)},{''.join(b_labels)}->{''.join(out_labels)}"
    from lucid.ops.einops import einsum
    return einsum(pattern, a, b)


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
    elif isinstance(axes, (tuple, list)) and len(axes) == 2:
        x, y = axes
        if isinstance(x, int) and isinstance(y, int):
            axes_a, axes_b = [x], [y]
        else:
            axes_a, axes_b = list(x), list(y)
    else:
        raise TypeError(f"tensordot: bad axes spec {axes!r}")
    # Route through einsum so autograd is intact.  Build labels:
    #  - first ndim_a chars from "abc..." for a
    #  - reuse the same letter for each contracted pair
    #  - then continue letters for b's non-contracted axes
    if a.ndim + b.ndim > 26:
        raise NotImplementedError("tensordot: too many axes for label set")
    a_labels = list("abcdefghijklmnopqrstuvwxyz"[:a.ndim])
    used = set(a_labels)
    b_labels = [None] * b.ndim
    # Pair contracted labels.
    for ai, bi in zip(axes_a, axes_b):
        ai_p = ai if ai >= 0 else ai + a.ndim
        bi_p = bi if bi >= 0 else bi + b.ndim
        b_labels[bi_p] = a_labels[ai_p]
    # Fill remaining b labels with fresh letters.
    pool = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i in range(b.ndim):
        if b_labels[i] is None:
            ch = next(pool)
            while ch in used:
                ch = next(pool)
            b_labels[i] = ch
            used.add(ch)
    a_str = "".join(a_labels)
    b_str = "".join(b_labels)
    contracted = set(a_labels[ai if ai >= 0 else ai + a.ndim] for ai in axes_a)
    out_labels = (
        [c for c in a_labels if c not in contracted]
        + [c for c in b_labels if c not in contracted]
    )
    pattern = f"{a_str},{b_str}->{''.join(out_labels)}"
    from lucid.ops.einops import einsum
    return einsum(pattern, a, b)


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
