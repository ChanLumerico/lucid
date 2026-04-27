"""
lucid.ops.utils — shape and indexing utilities (mirrors `lucid_legacy/_utils/`).

Includes: reshape, squeeze, unsqueeze, expand_dims, ravel, flatten,
broadcast_to, expand, stack, hstack, vstack, concatenate, split, chunk,
unbind, repeat, tile, pad, tril, triu, where, masked_fill, roll, gather,
diagonal, sort, argsort, argmin, argmax, nonzero, unique, topk,
histogram, histogram2d, histogramdd, meshgrid.
"""

from __future__ import annotations

from typing import Sequence

from lucid._C import engine as _C_engine
from lucid._tensor import Tensor
from lucid._bridge import impl_of, normalize_shape
from lucid.types import _ArrayLikeInt, _Scalar, _ShapeLike


__all__ = [
    "reshape", "squeeze", "unsqueeze", "expand_dims", "ravel",
    "stack", "hstack", "vstack", "concatenate",
    "split", "chunk", "unbind",
    "pad", "repeat", "tile",
    "flatten", "meshgrid",
    "tril", "triu", "broadcast_to", "expand",
    "masked_fill", "roll",
    "gather", "diagonal",
    "sort", "argsort", "argmin", "argmax",
    "nonzero", "unique", "topk",
    "where",
    "histogram", "histogram2d", "histogramdd",
]


# --------------------------------------------------------------------------- #
# View
# --------------------------------------------------------------------------- #

def reshape(a: Tensor, /, shape: _ShapeLike) -> Tensor:
    return Tensor._wrap(_C_engine.reshape(impl_of(a),
                                           normalize_shape(shape)))


def squeeze(a: Tensor, /, axis=None) -> Tensor:
    if axis is None:
        return Tensor._wrap(_C_engine.squeeze_all(impl_of(a)))
    if isinstance(axis, (list, tuple)):
        out = a
        for ax in sorted(axis, reverse=True):
            out = Tensor._wrap(_C_engine.squeeze(impl_of(out), int(ax)))
        return out
    return Tensor._wrap(_C_engine.squeeze(impl_of(a), int(axis)))


def unsqueeze(a: Tensor, /, axis) -> Tensor:
    if isinstance(axis, (list, tuple)):
        out = a
        for ax in sorted(axis):
            out = Tensor._wrap(_C_engine.unsqueeze(impl_of(out), int(ax)))
        return out
    return Tensor._wrap(_C_engine.unsqueeze(impl_of(a), int(axis)))


def expand_dims(a: Tensor, /, axis) -> Tensor:
    return unsqueeze(a, axis)


def ravel(a: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.ravel(impl_of(a)))


def flatten(a: Tensor, /, start_axis: int = 0, end_axis: int = -1) -> Tensor:
    return Tensor._wrap(_C_engine.flatten(
        impl_of(a), int(start_axis), int(end_axis)))


def broadcast_to(a: Tensor, /, shape: _ShapeLike) -> Tensor:
    return Tensor._wrap(_C_engine.broadcast_to(
        impl_of(a), normalize_shape(shape)))


def expand(a: Tensor, /, *sizes: int | _ShapeLike) -> Tensor:
    return broadcast_to(a, normalize_shape(sizes))


# --------------------------------------------------------------------------- #
# Concatenation / stacking / splitting
# --------------------------------------------------------------------------- #

def _impls(arr: Sequence[Tensor]) -> list:
    return [impl_of(t) for t in arr]


def stack(arr: Sequence[Tensor], /, axis: int = 0) -> Tensor:
    return Tensor._wrap(_C_engine.stack(_impls(arr), int(axis)))


def hstack(arr: Sequence[Tensor], /) -> Tensor:
    return Tensor._wrap(_C_engine.hstack(_impls(arr)))


def vstack(arr: Sequence[Tensor], /) -> Tensor:
    return Tensor._wrap(_C_engine.vstack(_impls(arr)))


def concatenate(arr: Sequence[Tensor], /, axis: int = 0) -> Tensor:
    return Tensor._wrap(_C_engine.concatenate(_impls(arr), int(axis)))


def split(
    a: Tensor, /,
    indices_or_sections: int | _ShapeLike,
    axis: int = 0,
) -> tuple[Tensor, ...]:
    if isinstance(indices_or_sections, int):
        pieces = _C_engine.split(impl_of(a), int(indices_or_sections), int(axis))
    else:
        pieces = _C_engine.split_at(impl_of(a), list(indices_or_sections), int(axis))
    return tuple(Tensor._wrap(p) for p in pieces)


def chunk(a: Tensor, /, chunks: int, axis: int = 0) -> tuple[Tensor, ...]:
    pieces = _C_engine.chunk(impl_of(a), int(chunks), int(axis))
    return tuple(Tensor._wrap(p) for p in pieces)


def unbind(a: Tensor, /, axis: int = 0) -> tuple[Tensor, ...]:
    pieces = _C_engine.unbind(impl_of(a), int(axis))
    return tuple(Tensor._wrap(p) for p in pieces)


# --------------------------------------------------------------------------- #
# Repetition / padding
# --------------------------------------------------------------------------- #

def repeat(a: Tensor, /, repeats: int, axis: int = 0) -> Tensor:
    return Tensor._wrap(_C_engine.repeat(impl_of(a), int(repeats), int(axis)))


def tile(a: Tensor, /, reps: int | Sequence[int]) -> Tensor:
    if isinstance(reps, int):
        reps = [reps]
    return Tensor._wrap(_C_engine.tile(impl_of(a), list(reps)))


def pad(a: Tensor, /, pad_width: _ArrayLikeInt, constant: float = 0.0) -> Tensor:
    # pad_width can be: int, (lo, hi), or [(lo, hi), ...]
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * a.ndim
    elif isinstance(pad_width, tuple) and len(pad_width) == 2 \
            and all(isinstance(x, int) for x in pad_width):
        pad_width = [pad_width] * a.ndim
    pw = [(int(lo), int(hi)) for lo, hi in pad_width]
    return Tensor._wrap(_C_engine.pad(impl_of(a), pw, float(constant)))


# --------------------------------------------------------------------------- #
# Triangular
# --------------------------------------------------------------------------- #

def tril(a: Tensor, /, diagonal: int = 0) -> Tensor:
    return Tensor._wrap(_C_engine.tril(impl_of(a), int(diagonal)))


def triu(a: Tensor, /, diagonal: int = 0) -> Tensor:
    return Tensor._wrap(_C_engine.triu(impl_of(a), int(diagonal)))


# --------------------------------------------------------------------------- #
# Selection / gathering
# --------------------------------------------------------------------------- #

def where(condition: Tensor, a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.where(
        impl_of(condition), impl_of(a), impl_of(b)))


def masked_fill(a: Tensor, /, mask: Tensor, value: _Scalar) -> Tensor:
    return Tensor._wrap(_C_engine.masked_fill(
        impl_of(a), impl_of(mask), float(value)))


def roll(
    a: Tensor, /,
    shifts: int | Sequence[int],
    axis: int | Sequence[int] | None = None,
) -> Tensor:
    if axis is None:
        # numpy semantics: flatten, roll, unflatten.
        flat = ravel(a)
        if isinstance(shifts, int):
            shifts = [shifts]
        out = Tensor._wrap(_C_engine.roll(impl_of(flat),
                                           list(shifts), [0]))
        return reshape(out, list(a.shape))
    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(axis, int):
        axis = [axis]
    return Tensor._wrap(_C_engine.roll(impl_of(a), list(shifts), list(axis)))


def gather(a: Tensor, /, axis: int, index: Tensor) -> Tensor:
    return Tensor._wrap(_C_engine.gather(impl_of(a), impl_of(index), int(axis)))


def diagonal(
    a: Tensor, /, offset: int = 0, axis1: int = 0, axis2: int = 1,
) -> Tensor:
    return Tensor._wrap(_C_engine.diagonal(
        impl_of(a), int(offset), int(axis1), int(axis2)))


# --------------------------------------------------------------------------- #
# Sort / search
# --------------------------------------------------------------------------- #

def sort(a: Tensor, /, axis: int = -1, descending: bool = False) -> Tensor:
    out = Tensor._wrap(_C_engine.sort(impl_of(a), int(axis)))
    if descending:
        # Reverse along axis.
        import numpy as np
        np_arr = out.numpy()
        return Tensor(np.flip(np_arr, axis=axis).copy(),
                      dtype=a.dtype, device=a.device)
    return out


def argsort(a: Tensor, /, axis: int = -1, descending: bool = False) -> Tensor:
    out = Tensor._wrap(_C_engine.argsort(impl_of(a), int(axis)))
    if descending:
        import numpy as np
        np_arr = out.numpy()
        return Tensor(np.flip(np_arr, axis=axis).copy(),
                      device=a.device)
    return out


def argmin(a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    if axis is None:
        # Flat min over all elements.
        return Tensor._wrap(_C_engine.argmin(impl_of(ravel(a)), 0,
                                              bool(keepdims)))
    return Tensor._wrap(_C_engine.argmin(impl_of(a), int(axis), bool(keepdims)))


def argmax(a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    if axis is None:
        return Tensor._wrap(_C_engine.argmax(impl_of(ravel(a)), 0,
                                              bool(keepdims)))
    return Tensor._wrap(_C_engine.argmax(impl_of(a), int(axis), bool(keepdims)))


def nonzero(a: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.nonzero(impl_of(a)))


def unique(a: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_engine.unique(impl_of(a)))


def topk(a: Tensor, /, k: int, axis: int = -1) -> Tensor:
    return Tensor._wrap(_C_engine.topk(impl_of(a), int(k), int(axis)))


# --------------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------------- #

def histogram(
    a: Tensor, /, bins: int = 10,
    range: tuple[float, float] | None = None,
    density: bool = False,
) -> tuple[Tensor, Tensor]:
    if a.ndim != 1:
        raise ValueError("histogram() expects a 1D tensor input.")
    if range is None:
        import numpy as np
        arr = a.numpy()
        range = (float(arr.min()), float(arr.max()))
    counts, edges = _C_engine.histogram(
        impl_of(a), int(bins), float(range[0]), float(range[1]), bool(density))
    return Tensor._wrap(counts), Tensor._wrap(edges)


def histogram2d(
    a: Tensor, b: Tensor, /,
    bins: list[int] | tuple[int, int] = (10, 10),
    range: list[tuple[float, float]] | None = None,
    density: bool = False,
) -> tuple[Tensor, Tensor]:
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    if range is None:
        import numpy as np
        ar, br = a.numpy(), b.numpy()
        range = [(float(ar.min()), float(ar.max())),
                 (float(br.min()), float(br.max()))]
    counts, edges = _C_engine.histogram2d(
        impl_of(a), impl_of(b),
        int(bins[0]), int(bins[1]),
        float(range[0][0]), float(range[0][1]),
        float(range[1][0]), float(range[1][1]),
        bool(density))
    return Tensor._wrap(counts), Tensor._wrap(edges)


def histogramdd(
    a: Tensor, /,
    bins: int | list[int],
    range: list[tuple[float, float]],
    density: bool = False,
) -> tuple[Tensor, Tensor]:
    if isinstance(bins, int):
        bins = [bins] * a.shape[1]
    counts, edges = _C_engine.histogramdd(
        impl_of(a),
        [int(b) for b in bins],
        [(float(lo), float(hi)) for lo, hi in range],
        bool(density))
    return Tensor._wrap(counts), Tensor._wrap(edges)


# --------------------------------------------------------------------------- #
# Meshgrid
# --------------------------------------------------------------------------- #

def meshgrid(
    *tensors: Tensor,
    indexing: str = "ij",
) -> tuple[Tensor, ...]:
    out = _C_engine.meshgrid([impl_of(t) for t in tensors],
                              indexing == "xy")
    return tuple(Tensor._wrap(o) for o in out)
