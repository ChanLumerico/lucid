"""The dim rule on the Python side: the engine's ``wrap_dim``, once.

Every dim argument the engine takes goes through its ``wrap_dim``: a tensor
of rank ``n`` answers to ``[-max(n, 1), max(n, 1))``, so a 0-d tensor has
one notional axis, named by both 0 and -1.  A Python wrapper that reads
``shape[dim]``, builds a permutation, or checks a dim list before the
engine sees it has to resolve the dim by the same rule.  Copies of it that
used the raw rank indexed past a 0-d shape (``list index out of range``) or
let an axis past the last one through.  This module is the only Python
copy: :func:`_to_axes` reads a dim spelling, :func:`_wrap_dim` and
:func:`_wrap_dims` resolve it, :func:`_dim_size` reads the axis's extent.
"""

import operator
from collections.abc import Sequence
from typing import SupportsIndex


def _to_axes(dim: object, name: str) -> list[int]:
    """``dim`` — ``None``, an index or a sequence of them — as engine axes."""
    if dim is None:
        return []
    if type(dim) is int:
        return [dim]
    # Plain ints in a tuple or list skip the protocol checks below, which
    # cost more than the reduction they guard.
    if type(dim) is tuple or type(dim) is list:
        axes = list(dim)
        for axis in axes:
            if type(axis) is not int:
                break
        else:
            return axes
    try:
        if isinstance(dim, SupportsIndex):
            return [operator.index(dim)]
        if isinstance(dim, Sequence) and not isinstance(dim, str):
            return [operator.index(d) for d in dim]
    except TypeError:
        pass  # an element that is not an index; refused below with the whole dim
    raise TypeError(
        f"{name}(): dim must be an int, a sequence of ints or None, got {dim!r}"
    )


def _wrap_dim(dim: int, ndim: int, op: str) -> int:
    """``dim`` counted from the front of a rank-``ndim`` tensor.

    A 0-d tensor's notional axis wraps to 0, which names no entry of its
    shape: read its extent with :func:`_dim_size`.

    Raises
    ------
    IndexError
        ``dim`` outside ``[-max(ndim, 1), max(ndim, 1))``.
    """
    extent = max(ndim, 1)
    wrapped = dim + extent if dim < 0 else dim
    if not 0 <= wrapped < extent:
        raise IndexError(
            f"{op}: axis out of range: got {dim}, expected "
            f"[{-extent}, {extent - 1}] for a tensor of rank {ndim}"
        )
    return wrapped


def _wrap_dims(dims: Sequence[int], ndim: int, op: str) -> list[int]:
    """Each of ``dims`` through :func:`_wrap_dim`, in order, none named twice.

    Raises
    ------
    IndexError
        Some dim is out of range.
    ValueError
        Two dims name the same axis (``0`` and ``-1`` of a 0-d tensor do).
    """
    wrapped = [_wrap_dim(d, ndim, op) for d in dims]
    if len(set(wrapped)) != len(wrapped):
        repeated = next(d for d in wrapped if wrapped.count(d) > 1)
        raise ValueError(
            f"{op}(): dim {repeated} appears multiple times in the list of "
            f"dims {tuple(dims)}"
        )
    return wrapped


def _dim_size(shape: Sequence[int], wrapped: int) -> int:
    """The extent of the wrapped axis: 1 for a 0-d tensor's notional one."""
    return int(shape[wrapped]) if shape else 1
