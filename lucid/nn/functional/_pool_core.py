"""
The bodies every ``nn.functional`` pool and pooling module runs.

``pooling.py`` holds the public signatures and their documentation; the
work lives here once: the engine call per rank, the unbatched round trip,
the argmax a caller asks for with ``return_indices``, and the fallback for
adaptive windows the engine cannot express.  The arguments arrive already
read by :mod:`lucid.nn.functional._spatial_args`.
"""

import itertools
import math
from collections.abc import Callable

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor.tensor import Tensor
from lucid._unsupported import unsupported_if
from lucid.nn.functional._spatial_args import (
    _PoolGeometry,
    _add_batch_axis,
    _drop_batch_axis,
    _ntuple,
)

_MAX_POOL: dict[int, Callable[..., _C_engine.TensorImpl]] = {
    1: _C_engine.nn.max_pool1d,
    2: _C_engine.nn.max_pool2d,
    3: _C_engine.nn.max_pool3d,
}
_AVG_POOL: dict[int, Callable[..., _C_engine.TensorImpl]] = {
    1: _C_engine.nn.avg_pool1d,
    2: _C_engine.nn.avg_pool2d,
    3: _C_engine.nn.avg_pool3d,
}
_ADAPTIVE_POOL: dict[tuple[str, int], Callable[..., _C_engine.TensorImpl]] = {
    ("mean", 1): _C_engine.nn.adaptive_avg_pool1d,
    ("mean", 2): _C_engine.nn.adaptive_avg_pool2d,
    ("mean", 3): _C_engine.nn.adaptive_avg_pool3d,
    ("max", 1): _C_engine.nn.adaptive_max_pool1d,
    ("max", 2): _C_engine.nn.adaptive_max_pool2d,
    ("max", 3): _C_engine.nn.adaptive_max_pool3d,
}


def _window_argmax_indices(
    x: Tensor, g: _PoolGeometry, out_spatial: tuple[int, ...]
) -> Tensor:
    """Where each pooling window's maximum sits, as the reference reports it.

    Per leading ``(N, C)`` slice, a flat index into the input's unpadded
    spatial extent — the form :func:`max_unpool2d` and its siblings take.
    Padding is ``-inf``, so it is never chosen, and a tie goes to the first
    element of the window in row-major order, as the reference's does.
    The values come from the engine kernel; this only locates them.
    """
    kernel, stride, padding = g.kernel, g.stride, g.padding
    n = len(kernel)
    lead = x.ndim - n
    spatial = tuple(int(s) for s in x.shape[lead:])
    pads: list[int] = []
    for d in reversed(range(n)):
        # Enough on the right to hold every window the engine produced,
        # including the partial last one ``ceil_mode`` keeps.
        right = (out_spatial[d] - 1) * stride[d] + kernel[d] - spatial[d] - padding[d]
        pads += [padding[d], max(right, 0)]
    windows = _lucid.nn.functional.pad(x, tuple(pads), value=float("-inf"))
    for d in range(n):
        windows = windows.unfold(lead + d, kernel[d], stride[d])
    for d in range(n):
        windows = windows.narrow(lead + d, 0, out_spatial[d])
    flat_window = windows.reshape(*x.shape[:lead], *out_spatial, math.prod(kernel))
    local = _lucid.argmax(flat_window, dim=-1)
    index = None
    for d in range(n):
        offset = (local // math.prod(kernel[d + 1 :])) % kernel[d]
        shape = [1] * n
        shape[d] = out_spatial[d]
        origin = _lucid.arange(out_spatial[d], dtype=_lucid.int64, device=x.device)
        coord = offset + (origin * stride[d] - padding[d]).reshape(*shape)
        index = coord if index is None else index * spatial[d] + coord
    assert index is not None
    return index


def _adaptive_ranges(n_in: int, n_out: int) -> list[tuple[int, int]]:
    """The reference's adaptive windows: ``[floor(i·in/out), ceil((i+1)·in/out))``."""
    return [((i * n_in) // n_out, -(-(i + 1) * n_in // n_out)) for i in range(n_out)]


def _adaptive_cells(
    x: Tensor, out_spatial: tuple[int, ...]
) -> list[tuple[tuple[tuple[int, int], ...], Tensor]]:
    """Every adaptive window of ``x``, row-major, with its ``(lo, hi)`` per axis."""
    n = len(out_spatial)
    lead = x.ndim - n
    spatial = tuple(int(s) for s in x.shape[lead:])
    cells = []
    for cell in itertools.product(
        *(_adaptive_ranges(i, o) for i, o in zip(spatial, out_spatial, strict=True))
    ):
        window = x
        for d, (lo, hi) in enumerate(cell):
            window = window.narrow(lead + d, lo, hi - lo)
        cells.append((cell, window))
    return cells


def _adaptive_argmax_indices(x: Tensor, out_spatial: tuple[int, ...]) -> Tensor:
    """:func:`_window_argmax_indices` for adaptive pooling's uneven windows."""
    n = len(out_spatial)
    lead = x.ndim - n
    spatial = tuple(int(s) for s in x.shape[lead:])
    if all(i % o == 0 for i, o in zip(spatial, out_spatial, strict=True)):
        kernel = tuple(i // o for i, o in zip(spatial, out_spatial, strict=True))
        g = _PoolGeometry(kernel, kernel, (0,) * n, (1,) * n)
        return _window_argmax_indices(x, g, out_spatial)
    found: list[Tensor] = []
    for cell, window in _adaptive_cells(x, out_spatial):
        extent = [hi - lo for lo, hi in cell]
        local = _lucid.argmax(window.reshape(*x.shape[:lead], -1), dim=-1)
        index = None
        for d in range(n):
            coord = (local // math.prod(extent[d + 1 :])) % extent[d] + cell[d][0]
            index = coord if index is None else index * spatial[d] + coord
        assert index is not None
        found.append(index)
    return _lucid.stack(found, dim=-1).reshape(*x.shape[:lead], *out_spatial)


def _adaptive_pool_python(
    x: Tensor, output_size: tuple[int, ...], reduce: str
) -> Tensor:
    """Engine fallback for adaptive pooling with non-divisible sizes.

    Reduces each window ``[floor(i·in/out), ceil((i+1)·in/out))`` per axis,
    the reference's contract.  Each window is a ``narrow`` and a reduction,
    engine ops both, so the result stays on the input's device with no
    host round-trip.  ``reduce`` is ``"mean"`` for average pooling and
    ``"max"`` for max pooling; the window geometry is the same for both.
    """
    n = len(output_size)
    lead = x.ndim - n
    dims = tuple(range(lead, x.ndim))
    values = [
        window.max(dim=dims) if reduce == "max" else window.mean(dim=dims)
        for _, window in _adaptive_cells(x, output_size)
    ]
    return _lucid.stack(values, dim=-1).reshape(*x.shape[:lead], *output_size)


def _max_pool(
    x: Tensor, g: _PoolGeometry, ceil_mode: bool, return_indices: bool
) -> Tensor | tuple[Tensor, Tensor]:
    """The one body of ``max_pool{1,2,3}d`` and their modules."""
    n = len(g.kernel)
    unsupported_if(
        any(d != 1 for d in g.dilation),
        f"max_pool{n}d",
        "dilation",
        g.dilation,
        detail="Only dilation=1 is supported.",
    )
    xb, unbatched = _add_batch_axis(x, n)
    out = _wrap(_MAX_POOL[n](_unwrap(xb), *g.kernel, *g.stride, *g.padding, ceil_mode))
    if not return_indices:
        return _drop_batch_axis(out, unbatched)
    index = _window_argmax_indices(xb, g, tuple(int(v) for v in out.shape[-n:]))
    return _drop_batch_axis(out, unbatched), _drop_batch_axis(index, unbatched)


def _avg_pool(
    x: Tensor,
    g: _PoolGeometry,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> Tensor:
    """The one body of ``avg_pool{1,2,3}d`` and their modules."""
    n = len(g.kernel)
    unsupported_if(
        divisor_override is not None,
        f"avg_pool{n}d",
        "divisor_override",
        divisor_override,
        detail="The window size is always the divisor.",
    )
    xb, unbatched = _add_batch_axis(x, n)
    out = _wrap(
        _AVG_POOL[n](
            _unwrap(xb), *g.kernel, *g.stride, *g.padding, ceil_mode, count_include_pad
        )
    )
    return _drop_batch_axis(out, unbatched)


def _adaptive_pool(
    x: Tensor, output_size: int | tuple[int, ...], n: int, reduce: str
) -> Tensor:
    """The one body of the ``adaptive_{avg,max}_pool{1,2,3}d`` values.

    The engine op reduces adaptive pooling to a fixed-kernel pool, which
    only expresses the case where every input axis divides its output
    evenly; :func:`_adaptive_pool_python` covers the rest.  Avg and max both
    route through here, so they cannot drift apart on which sizes they take.
    """
    op = f"adaptive_{'avg' if reduce == 'mean' else 'max'}_pool{n}d"
    xb, unbatched = _add_batch_axis(x, n)
    sizes = _ntuple(output_size, n, op, "output_size")
    spatial = tuple(int(s) for s in xb.shape[-n:])
    if all(i % o == 0 for i, o in zip(spatial, sizes, strict=True)):
        out = _wrap(_ADAPTIVE_POOL[(reduce, n)](_unwrap(xb), *sizes))
    else:
        out = _adaptive_pool_python(xb, sizes, reduce)
    return _drop_batch_axis(out, unbatched)


def _adaptive_max_pool(
    x: Tensor, output_size: int | tuple[int, ...], n: int, return_indices: bool
) -> Tensor | tuple[Tensor, Tensor]:
    """:func:`_adaptive_pool` for max, with the argmax when it is asked for."""
    out = _adaptive_pool(x, output_size, n, "max")
    if not return_indices:
        return out
    xb, unbatched = _add_batch_axis(x, n)
    sizes = tuple(int(v) for v in out.shape[-n:])
    return out, _drop_batch_axis(_adaptive_argmax_indices(xb, sizes), unbatched)
