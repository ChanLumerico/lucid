"""Where a view sits in its base's storage, worked out from the op that took it.

A Metal view is an MLX array of its own, so it has no strides into its
base's buffer to read back.  :mod:`lucid._tensor._metal_views` gives it the
layout the reference gives the same view — strides and the first element's
place in the storage, in elements — from the op's own arguments: one rule
per view op, composed along the chain of views.  Everything here is
arithmetic on shapes and strides; nothing reads a tensor.
"""

import operator
from collections.abc import Callable, Sequence
from typing import NamedTuple, SupportsIndex

#: A shape or a stride, in elements.
type _Dims = tuple[int, ...]


class _Layout(NamedTuple):
    """Where a view's elements sit in its storage, in elements, as in the reference."""

    stride: _Dims
    offset: int


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        stride[d] = stride[d + 1] * max(shape[d + 1], 1)
    return tuple(stride)


def _is_contiguous_stride(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    if 0 in shape:
        return True  # no element to be out of place
    expected = 1
    for d in range(len(shape) - 1, -1, -1):
        if shape[d] != 1:
            if stride[d] != expected:
                return False
            expected *= shape[d]
    return True


def _squeezes_only(
    in_shape: tuple[int, ...],
    in_stride: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    sizes = [s for s in in_shape if s != 1]
    kept = [st for s, st in zip(in_shape, in_stride, strict=True) if s != 1]
    want = [s for s in out_shape if s != 1]
    if want != sizes:
        return None
    stride = [0] * len(out_shape)
    k = len(kept)
    next_st = 1
    for d in range(len(out_shape) - 1, -1, -1):
        if out_shape[d] != 1:
            k -= 1
            stride[d] = kept[k]
        else:
            stride[d] = next_st
        next_st = stride[d] * out_shape[d]
    return tuple(stride)


def _fill_chunk(
    stride: list[int],
    out_shape: tuple[int, ...],
    view_d: int,
    in_numel: int,
    chunk_stride: int,
) -> tuple[int, int]:
    view_numel = 1
    while view_d >= 0 and (view_numel < in_numel or out_shape[view_d] == 1):
        stride[view_d] = view_numel * chunk_stride
        view_numel *= out_shape[view_d]
        view_d -= 1
    return view_d, view_numel


def _chunk_strides(
    in_shape: tuple[int, ...],
    in_stride: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    if not in_shape:
        return (1,) * len(out_shape)
    if 0 in in_shape:
        return in_stride if in_shape == out_shape else _contiguous_stride(out_shape)
    stride = [0] * len(out_shape)
    view_d = len(out_shape) - 1
    chunk_stride = in_stride[-1]
    in_numel = 1
    for k in range(len(in_shape) - 1, -1, -1):
        if in_shape[k] == 1 and k > 0 and in_numel == 1:
            chunk_stride = in_stride[k - 1]
            continue
        in_numel *= in_shape[k]
        if k > 0 and (
            in_shape[k - 1] == 1 or in_stride[k - 1] == in_numel * chunk_stride
        ):
            continue
        view_d, view_numel = _fill_chunk(
            stride, out_shape, view_d, in_numel, chunk_stride
        )
        if view_numel != in_numel:
            return None
        if k > 0:
            chunk_stride = in_stride[k - 1]
            in_numel = 1
    return tuple(stride) if view_d == -1 else None


def _view_strides(
    in_shape: tuple[int, ...],
    in_stride: tuple[int, ...],
    out_shape: tuple[int, ...],
    squeeze: bool,
) -> tuple[int, ...] | None:
    if squeeze:
        return _squeezes_only(in_shape, in_stride, out_shape)
    # An empty input keeps its strides when its shape is kept, as the
    # engine's chunk rule has it.
    if 0 not in in_shape and _is_contiguous_stride(in_shape, in_stride):
        return _contiguous_stride(out_shape)
    return _chunk_strides(in_shape, in_stride, out_shape)


# ── One rule per view op ─────────────────────────────────────────────────────
#
# A later ``view`` of a traced view needs its strides: the merge-heads
# ``attn.permute(0, 2, 1, 3).reshape(b, n, e)`` asks of every attention layer,
# every step.  ``as_strided``, ``stride()`` and ``storage_offset()`` need its
# first element's place in the storage too.  Each rule below works both out
# from the op's own arguments, matching the CPU allocation frame (LCD-517) —
# replaying the op on a CPU tensor to read them back would cost a host
# allocation and a CPU op per step.
# The strides agree with the reference on every axis that moves an element;
# an axis of length one moves none, and the stride it is given varies with
# how the view was composed.  A rule takes the view's geometry, then the
# parameters of the method it replays, which the op has already checked.


class _Geometry(NamedTuple):
    shape: _Dims  # the input's
    stride: _Dims  # the input's
    out: _Dims  # the view's shape
    offset: int = 0  # the input's first element, in its storage
    piece: int = 0  # which of split / chunk / unbind's pieces the view is


def _dim(d: SupportsIndex, ndim: int) -> int:
    i = operator.index(d)
    return i + ndim if i < 0 else i


def _axes(value: SupportsIndex | Sequence[SupportsIndex]) -> list[int]:
    if isinstance(value, Sequence):
        return [operator.index(v) for v in value]
    return [operator.index(value)]


def _swapped(stride: _Dims, a: SupportsIndex, b: SupportsIndex) -> _Dims:
    if not stride:
        return ()
    out = list(stride)
    i, j = _dim(a, len(out)), _dim(b, len(out))
    out[i], out[j] = out[j], out[i]
    return tuple(out)


def _reversed(g: _Geometry, /) -> _Layout:
    # ``t()``, ``T`` and an axis-less ``transpose()`` reverse every axis.
    return _Layout(g.stride[::-1], g.offset)


def _last_two_swapped(g: _Geometry, /) -> _Layout:
    return _Layout(_swapped(g.stride, -2, -1), g.offset)


def _transpose_rule(
    g: _Geometry,
    /,
    dim0: SupportsIndex | None = None,
    dim1: SupportsIndex | None = None,
) -> _Layout:
    if dim0 is None or dim1 is None:
        return _reversed(g)
    return _Layout(_swapped(g.stride, dim0, dim1), g.offset)


def _swapaxes_rule(
    g: _Geometry, /, axis0: SupportsIndex, axis1: SupportsIndex
) -> _Layout:
    return _Layout(_swapped(g.stride, axis0, axis1), g.offset)


def _permute_rule(
    g: _Geometry,
    /,
    *dims: SupportsIndex | Sequence[SupportsIndex],
    **named: SupportsIndex | Sequence[SupportsIndex],
) -> _Layout:
    # ``permute(2, 0, 1)``, ``permute((2, 0, 1))`` and ``permute(dims=...)``.
    given = (named["dims"],) if "dims" in named else dims
    stride = tuple(
        g.stride[_dim(d, len(g.stride))] for part in given for d in _axes(part)
    )
    return _Layout(stride, g.offset)


def _movedim_rule(
    g: _Geometry,
    /,
    source: SupportsIndex | Sequence[SupportsIndex],
    destination: SupportsIndex | Sequence[SupportsIndex],
) -> _Layout:
    n = len(g.stride)
    if not n:
        return _Layout((), g.offset)
    moved = [_dim(s, n) for s in _axes(source)]
    order = [-1] * n
    for s, d in zip(moved, _axes(destination), strict=True):
        order[_dim(d, n)] = s
    rest = iter(d for d in range(n) if d not in moved)
    stride = tuple(g.stride[next(rest) if o == -1 else o] for o in order)
    return _Layout(stride, g.offset)


def _cut(g: _Geometry, dim: SupportsIndex, start: int) -> _Layout:
    """A run along ``dim`` from ``start``: the steps stay, the first element moves."""
    return _Layout(g.stride, g.offset + start * g.stride[_dim(dim, len(g.stride))])


def _narrow_rule(
    g: _Geometry, /, dim: SupportsIndex, start: SupportsIndex, length: SupportsIndex
) -> _Layout:
    del length  # the run's length is the view's shape
    s = operator.index(start)
    return _cut(g, dim, s + g.shape[_dim(dim, len(g.shape))] if s < 0 else s)


def _split_rule(
    g: _Geometry,
    /,
    split_size: SupportsIndex | Sequence[SupportsIndex],
    dim: SupportsIndex = 0,
) -> _Layout:
    if isinstance(split_size, Sequence):
        start = sum(operator.index(n) for n in split_size[: g.piece])
    else:
        start = g.piece * operator.index(split_size)
    return _cut(g, dim, start)


def _chunk_rule(
    g: _Geometry, /, chunks: SupportsIndex, dim: SupportsIndex = 0
) -> _Layout:
    # Every chunk but the last is the ceiling share of the axis.
    length = g.shape[_dim(dim, len(g.shape))]
    return _cut(g, dim, g.piece * -(-length // operator.index(chunks)))


def _unbind_rule(g: _Geometry, /, dim: SupportsIndex = 0) -> _Layout:
    d = _dim(dim, len(g.stride))
    stride = g.stride[:d] + g.stride[d + 1 :]
    return _Layout(stride, g.offset + g.piece * g.stride[d])


def _diagonal_rule(
    g: _Geometry,
    /,
    offset: SupportsIndex = 0,
    dim1: SupportsIndex = -2,
    dim2: SupportsIndex = -1,
) -> _Layout:
    i, j = _dim(dim1, len(g.stride)), _dim(dim2, len(g.stride))
    rest = tuple(s for d, s in enumerate(g.stride) if d not in (i, j))
    k = operator.index(offset)
    # The reference starts an empty diagonal where its input starts.
    start = 0 if not g.out[-1] else k * g.stride[j] if k >= 0 else -k * g.stride[i]
    return _Layout((*rest, g.stride[i] + g.stride[j]), g.offset + start)


def _unfold_rule(
    g: _Geometry, /, dimension: SupportsIndex, size: SupportsIndex, step: SupportsIndex
) -> _Layout:
    del size  # the window's length is the view's last axis
    if not g.stride:
        return _Layout((1,), g.offset)  # a 0-d input is one window of its one element
    d = _dim(dimension, len(g.stride))
    st = g.stride
    return _Layout(
        (*st[:d], st[d] * operator.index(step), *st[d + 1 :], st[d]), g.offset
    )


def _expanded(g: _Geometry, /, *args: object, **kwargs: object) -> _Layout:
    # ``expand``, ``expand_as`` and ``broadcast_to``: a new leading axis, or
    # one stretched from length one, repeats an element.
    del args, kwargs  # the view's shape already says where it repeats
    lead = len(g.out) - len(g.shape)
    kept = (
        st if n == m else 0
        for n, st, m in zip(g.shape, g.stride, g.out[lead:], strict=True)
    )
    return _Layout((*(0,) * lead, *kept), g.offset)


def _index_rule(g: _Geometry, /, key: object) -> _Layout:
    """Basic indexing: an int drops its axis, a slice steps along it, ``None`` adds one.

    Only a key the CPU reads in place is linked (``_metal_views._is_view_key``),
    so every slice steps forwards.
    """
    parts = key if isinstance(key, tuple) else (key,)
    taken = sum(1 for part in parts if part is not None and part is not Ellipsis)
    found: list[int | None] = []
    offset = g.offset
    d = 0
    for part in parts:
        if part is None:
            found.append(None)
        elif part is Ellipsis:
            span = len(g.stride) - taken
            found.extend(g.stride[d : d + span])
            d += span
        elif isinstance(part, slice):
            start, _, step = part.indices(g.shape[d])
            offset += start * g.stride[d]
            found.append(g.stride[d] * step)
            d += 1
        elif isinstance(part, (int, SupportsIndex)):
            i = operator.index(part)
            offset += (i + g.shape[d] if i < 0 else i) * g.stride[d]
            d += 1
    found.extend(g.stride[d:])
    # An inserted axis moves no element; like the engine's unsqueeze it takes
    # the extent of the axis inside it.
    result = [0] * len(found)
    inner = 1
    for i in range(len(found) - 1, -1, -1):
        got = found[i]
        result[i] = inner if got is None else got
        inner = result[i] * g.out[i]
    return _Layout(tuple(result), offset)


#: Every op a traced view replays, by the name it replays under.
_LAYOUT_RULES: dict[str, Callable[..., _Layout]] = {
    "transpose": _transpose_rule,
    "swapaxes": _swapaxes_rule,
    "permute": _permute_rule,
    "movedim": _movedim_rule,
    "t": _reversed,
    "T": _reversed,
    "mT": _last_two_swapped,
    "narrow": _narrow_rule,
    "split": _split_rule,
    "chunk": _chunk_rule,
    "unbind": _unbind_rule,
    "diagonal": _diagonal_rule,
    "unfold": _unfold_rule,
    "expand": _expanded,
    "expand_as": _expanded,
    "broadcast_to": _expanded,
    "__getitem__": _index_rule,
}


def _dense_order(shape: _Dims, stride: _Dims) -> list[int] | None:
    """The axis order that walks a dense block in storage order, or ``None``.

    A block is dense when its elements are one gapless run of the storage in
    some order of its axes: ``y.t()`` is, ``y[:, ::2]`` and ``y.diagonal()``
    are not.
    """
    if 0 in shape:
        return None
    moving = sorted(
        (d for d, n in enumerate(shape) if n != 1), key=lambda d: -stride[d]
    )
    expected = 1
    for d in reversed(moving):
        if stride[d] != expected:
            return None
        expected *= shape[d]
    return [d for d, n in enumerate(shape) if n == 1] + moving
