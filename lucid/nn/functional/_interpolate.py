"""
Resampling for :func:`~lucid.nn.functional.interpolate`, one axis at a time.

Every mode ``interpolate`` accepts is separable: an output sample along
one axis reads a few input samples along that axis, with weights that
depend only on its position.  So each axis is described by its *taps* —
for every tap, the input index each output sample reads and the weight
it reads it with.  An axis is applied as a sum of gathers, or — while it
is short enough for the dense ``(out, in)`` matrix to cost less — as one
matrix product; the plan for each configuration is cached.

Where an output sample lands in the input is decided in one place,
:func:`_coordinate_scale` and :func:`_source_index`, which follow the
reference framework's rules: with ``align_corners`` the end samples are
pinned together; otherwise the step is ``1 / scale_factor`` whenever a
scale factor was given (and not recomputed), and ``in / out`` only when
the size was.  The two differ whenever ``in * scale_factor`` is not a
whole number, and that difference is the whole of what used to be wrong.

The engine kernels take sizes only and step by ``in / out``.  They run
when that is the same mapping — a size, a recomputed or whole-number
scale — and the gathers run when it is not.
"""

import dataclasses
import functools
import math
import struct
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional._grid_sampling import cubic_coefficients
from lucid.nn.functional.pooling import (
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# The spatial ranks each mode resamples.
_RANKS: dict[str, tuple[int, ...]] = {
    "nearest": (1, 2, 3),
    "nearest-exact": (1, 2, 3),
    "area": (1, 2, 3),
    "linear": (1,),
    "bilinear": (2,),
    "bicubic": (2,),
    "trilinear": (3,),
}
_LINEAR = frozenset({"linear", "bilinear", "trilinear"})
_INTERPOLATING = _LINEAR | {"bicubic"}
_ANTIALIASED = frozenset({"bilinear", "bicubic"})

# Each tap: the input index every output sample reads, and its weight
# (``None`` when the only tap has weight one — nearest).
type _Taps = list[tuple[list[int], list[float] | None]]
type _FrozenTaps = tuple[tuple[tuple[int, ...], tuple[float, ...] | None], ...]


def _f32(value: float) -> float:
    """``value`` rounded to single precision, as the reference's nearest
    kernel computes its source index in ``float``."""
    rounded: float = struct.unpack("f", struct.pack("f", value))[0]
    return rounded


def _coordinate_scale(
    in_size: int, out_size: int, align_corners: bool, scale: float | None
) -> float:
    """Distance in input samples between two neighbouring output samples."""
    if align_corners:
        return (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    if scale is not None and scale > 0:
        return 1.0 / scale
    return in_size / out_size


def _source_index(ratio: float, dst: int, align_corners: bool, cubic: bool) -> float:
    """Input coordinate output sample ``dst`` is centred on.

    Half-pixel centres unless the corners are aligned.  Linear filters
    clamp a coordinate left of the first sample onto it; the cubic
    filter keeps it, because its taps clamp instead.
    """
    if align_corners:
        return ratio * dst
    source = ratio * (dst + 0.5) - 0.5
    return source if cubic or source >= 0 else 0.0


def _nearest_indices(
    in_size: int, out_size: int, scale: float | None, exact: bool, rank: int
) -> list[int]:
    """Input index each output sample copies, for ``nearest[-exact]``."""
    last = in_size - 1
    if exact:
        ratio = _f32(_coordinate_scale(in_size, out_size, False, scale))
        return [min(math.floor(_f32((d + 0.5) * ratio)), last) for d in range(out_size)]
    # Plain nearest on a 2-D input pins these two cases whatever the scale
    # factor said; the reference keeps them for compatibility (and only in
    # its 2-D kernel), so they are the contract there and nowhere else.
    if rank == 2 and out_size == in_size:
        return list(range(out_size))
    if rank == 2 and out_size == 2 * in_size:
        return [d >> 1 for d in range(out_size)]
    ratio = _f32(_coordinate_scale(in_size, out_size, False, scale))
    return [min(math.floor(_f32(d * ratio)), last) for d in range(out_size)]


def _floor_and_fraction(source: float, in_size: int) -> tuple[int, float]:
    """The sample left of ``source`` (never past the last) and the distance to it."""
    base = min(math.floor(source), in_size - 1)
    return base, min(max(source - base, 0.0), 1.0)


def _linear_taps(
    in_size: int, out_size: int, align_corners: bool, scale: float | None
) -> _Taps:
    if out_size == in_size:
        return [(list(range(out_size)), None)]
    ratio = _coordinate_scale(in_size, out_size, align_corners, scale)
    low: list[int] = []
    high: list[int] = []
    upper: list[float] = []
    for d in range(out_size):
        base, frac = _floor_and_fraction(
            _source_index(ratio, d, align_corners, cubic=False), in_size
        )
        low.append(base)
        high.append(base + 1 if base < in_size - 1 else base)
        upper.append(frac)
    return [(low, [1.0 - f for f in upper]), (high, upper)]


def _cubic_taps(
    in_size: int, out_size: int, align_corners: bool, scale: float | None
) -> _Taps:
    ratio = _coordinate_scale(in_size, out_size, align_corners, scale)
    taps: _Taps = [([], []) for _ in range(4)]
    for d in range(out_size):
        base, frac = _floor_and_fraction(
            _source_index(ratio, d, align_corners, cubic=True), in_size
        )
        for tap, weight in enumerate(cubic_coefficients(frac)):
            indices, weights = taps[tap]
            indices.append(min(max(base + tap - 1, 0), in_size - 1))
            assert weights is not None
            weights.append(weight)
    return taps


def _triangle(x: float) -> float:
    return max(1.0 - abs(x), 0.0)


def _keys_cubic(x: float) -> float:
    # Keys' kernel with a = -0.5: the antialiasing filter is the smoother
    # member of the family, not the -0.75 one plain bicubic uses.
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    return 0.0


def _antialias_taps(
    in_size: int,
    out_size: int,
    align_corners: bool,
    scale: float | None,
    cubic: bool,
) -> _Taps:
    """Taps of the filter stretched over the input samples one output covers.

    Downsampling by ``r`` widens the filter ``r`` times, so every input
    sample contributes; upsampling leaves it at its own width.  Weights
    are normalised per output sample, which is what lets the stretched
    filter keep a constant image constant at the borders.  An axis that
    keeps its size is left alone whatever the scale factor said, as the
    reference does.
    """
    if out_size == in_size:
        return [(list(range(out_size)), None)]
    width = 4 if cubic else 2
    kernel: Callable[[float], float] = _keys_cubic if cubic else _triangle
    ratio = _coordinate_scale(in_size, out_size, align_corners, scale)
    support = width * 0.5 * ratio if ratio >= 1.0 else width * 0.5
    inverse = 1.0 / ratio if ratio >= 1.0 else 1.0
    count = math.ceil(support) * 2 + 1
    taps: _Taps = [([], []) for _ in range(count)]
    for d in range(out_size):
        centre = ratio * (d + 0.5)
        # ``int`` truncates toward zero, as the reference's cast does.
        start = max(int(centre - support + 0.5), 0)
        length = min(max(min(int(centre + support + 0.5), in_size) - start, 0), count)
        raw = [kernel((j + start - centre + 0.5) * inverse) for j in range(length)]
        total = sum(raw)
        normed = [w / total for w in raw] if total != 0.0 else raw
        for tap in range(count):
            indices, weights = taps[tap]
            assert weights is not None
            indices.append(min(start + tap, in_size - 1))
            weights.append(normed[tap] if tap < length else 0.0)
    return taps


# An axis is applied as a matrix product (out x in, built from the taps)
# while the input is at most this many times longer than the taps are
# many; past that, the dense product does more work than gathering does.
_DENSE_PER_TAP = 512


# Hashed by identity (``eq=False``): the plan cache hands out one object
# per configuration, so the matrix cache below can key on it without
# hashing every tap on every call.
@dataclasses.dataclass(frozen=True, eq=False)
class _AxisPlan:
    """How one axis is resampled — worked out once per configuration."""

    taps: _FrozenTaps
    identity: bool
    # Non-zero entries of the (out, in) resampling matrix, row-major,
    # when the axis is applied as a product.
    dense: tuple[tuple[int, ...], tuple[float, ...]] | None


def _is_identity(taps: _Taps, in_size: int) -> bool:
    out_size = len(taps[0][0])
    if out_size != in_size:
        return False
    for indices, weights in taps:
        for d, index in enumerate(indices):
            weight = 1.0 if weights is None else weights[d]
            if weight != (1.0 if index == d else 0.0):
                return False
    return True


def _dense_entries(
    taps: _Taps, in_size: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """The taps as matrix entries; taps clamped onto one sample add up."""
    entries: dict[int, float] = {}
    for indices, weights in taps:
        assert weights is not None
        for d, (index, weight) in enumerate(zip(indices, weights, strict=True)):
            if weight != 0.0:
                position = d * in_size + index
                entries[position] = entries.get(position, 0.0) + weight
    return tuple(entries), tuple(entries.values())


# Building a matrix from its entries is a host-to-device copy of every
# entry and a scatter, a third of the cost of a large resize; a pipeline
# resizing every image to one size rebuilds the same few.  Bounded, since
# each entry holds an (out, in) matrix on its device.
@functools.lru_cache(maxsize=16)
def _dense_matrix(
    plan: _AxisPlan, in_size: int, device: str, dtype: _lucid.dtype
) -> Tensor:
    assert plan.dense is not None
    positions, weights = plan.dense
    out_size = len(plan.taps[0][0])
    matrix = _lucid.zeros(out_size * in_size, dtype=dtype, device=device)
    matrix[_lucid.tensor(list(positions), dtype=_lucid.int64, device=device)] = (
        _lucid.tensor(list(weights), dtype=dtype, device=device)
    )
    return matrix.reshape(out_size, in_size)


def _apply_dense(x: Tensor, axis: int, plan: _AxisPlan) -> Tensor:
    matrix = _dense_matrix(plan, int(x.shape[axis]), x.device.type, x.dtype)
    # The two trailing axes multiply in place; any other is moved there.
    if axis == x.ndim - 1:
        return x @ matrix.mT
    if axis == x.ndim - 2:
        return matrix @ x
    return (matrix @ x.transpose(axis, -2)).transpose(axis, -2)


def _apply_gathers(x: Tensor, axis: int, taps: _FrozenTaps) -> Tensor:
    """Sum over taps of the gathered samples times their weights."""
    device = x.device.type
    shape = [1] * x.ndim
    shape[axis] = len(taps[0][0])
    total: Tensor | None = None
    for indices, weights in taps:
        if weights is not None and not any(weights):
            continue
        taken = x.index_select(
            axis, _lucid.tensor(list(indices), dtype=_lucid.int64, device=device)
        )
        if weights is not None:
            taken = taken * _lucid.tensor(
                list(weights), dtype=x.dtype, device=device
            ).reshape(tuple(shape))
        total = taken if total is None else total + taken
    assert total is not None
    return total


def _per_axis(
    name: str, value: float | Sequence[float], spatial: list[int]
) -> list[float]:
    """One entry per spatial axis: a number is repeated, a sequence checked."""
    if isinstance(value, (int, float)):
        return [float(value)] * len(spatial)
    entries = [float(v) for v in value]
    if len(entries) != len(spatial):
        raise ValueError(
            f"interpolate: {name} {tuple(value)} has {len(entries)} entries for an "
            f"input with {len(spatial)} spatial axes {tuple(spatial)}"
        )
    return entries


def _resolve(
    x: Tensor,
    size: int | Sequence[int] | None,
    scale_factor: float | Sequence[float] | None,
    recompute_scale_factor: bool | None,
) -> tuple[list[int], list[float | None]]:
    """Output size of each spatial axis, and the scale its mapping uses."""
    spatial = [int(s) for s in x.shape[2:]]
    if (size is None) == (scale_factor is None):
        raise ValueError(
            "interpolate: either size or scale_factor must be given, not both"
        )
    scales: list[float | None] = [None] * len(spatial)
    if size is not None:
        if recompute_scale_factor:
            raise ValueError(
                "interpolate: recompute_scale_factor is not meaningful with an "
                "explicit size"
            )
        sizes = [int(n) for n in _per_axis("size", size, spatial)]
    else:
        assert scale_factor is not None
        factors = _per_axis("scale_factor", scale_factor, spatial)
        sizes = [math.floor(n * f) for n, f in zip(spatial, factors, strict=True)]
        if not recompute_scale_factor:
            scales = list(factors)
    if min(sizes + spatial) <= 0:
        raise ValueError(
            f"interpolate: input {tuple(spatial)} and output {tuple(sizes)} spatial "
            "sizes must all be positive"
        )
    return sizes, scales


def _check_mode(
    x: Tensor, mode: str, align_corners: bool | None, antialias: bool
) -> None:
    if mode not in _RANKS:
        raise ValueError(
            f"interpolate: unknown mode {mode!r}; expected one of {sorted(_RANKS)}"
        )
    rank = x.ndim - 2
    if rank not in _RANKS[mode]:
        raise ValueError(
            f"interpolate: mode {mode!r} resamples {_RANKS[mode]} spatial axes, "
            f"got an input of shape {tuple(x.shape)} ({rank} spatial)"
        )
    if align_corners is not None and mode not in _INTERPOLATING:
        raise ValueError(
            "interpolate: align_corners can only be set with the interpolating "
            f"modes {sorted(_INTERPOLATING)}, not {mode!r}"
        )
    if antialias and mode not in _ANTIALIASED:
        raise ValueError(
            f"interpolate: antialias is only defined for {sorted(_ANTIALIASED)} "
            f"on 4-D input, not {mode!r}"
        )
    if mode != "nearest" and mode != "nearest-exact" and not x.is_floating_point():
        raise TypeError(
            f"interpolate: mode {mode!r} needs a floating-point input, got {x.dtype}"
        )


def _engine_resample(
    x: Tensor, mode: str, sizes: list[int], align_corners: bool
) -> Tensor:
    """The size-only kernel; a 1-D axis rides through as a unit-height plane."""
    impl = _unwrap(x)
    if len(sizes) == 1:
        impl = _C_engine.unsqueeze(impl, 2)
        sizes = [1, sizes[0]]
    nn = _C_engine.nn
    if len(sizes) == 2:
        h, w = sizes
        if mode == "nearest":
            out = nn.interpolate_nearest_2d(impl, h, w)
        else:
            out = nn.interpolate_bilinear(impl, h, w, align_corners)
    else:
        d, h, w = sizes
        if mode == "nearest":
            out = nn.interpolate_nearest_3d(impl, d, h, w)
        else:
            out = nn.interpolate_trilinear(impl, d, h, w, align_corners)
    return _wrap(_C_engine.squeeze(out, 2) if x.ndim == 3 else out)


def _engine_maps_like(
    mode: str,
    spatial: list[int],
    sizes: list[int],
    scales: list[float | None],
    align_corners: bool,
) -> bool:
    """Whether the size-only engine kernel lands every sample where the
    reference does.  The kernels step by ``in / out`` (or pin corners), on
    either device, by the reference's own rule for a size."""
    rank = len(spatial)
    for n, m, s in zip(spatial, sizes, scales, strict=True):
        if mode == "nearest":
            if s is None:
                continue
            reference = _nearest_indices(n, m, s, exact=False, rank=rank)
            if _nearest_indices(n, m, None, exact=False, rank=rank) != reference:
                return False
        elif not (align_corners or m == n or s is None):
            ratio = _coordinate_scale(n, m, False, s)
            if not math.isclose(ratio, n / m, rel_tol=1e-12):
                return False
    return True


@functools.lru_cache(maxsize=256)
def _axis_plan(
    mode: str,
    in_size: int,
    out_size: int,
    align_corners: bool,
    scale: float | None,
    antialias: bool,
    rank: int,
) -> _AxisPlan:
    """How to resample one axis, frozen so the cache can hand it out again.

    Building them is a Python loop over the output samples (and, for the
    stretched antialiasing filter, over every input sample each covers),
    which a data pipeline resizing every image to the same size would
    otherwise repeat per call.
    """
    if antialias:
        taps = _antialias_taps(
            in_size, out_size, align_corners, scale, cubic=mode == "bicubic"
        )
    elif mode in ("nearest", "nearest-exact"):
        exact = mode == "nearest-exact"
        taps = [(_nearest_indices(in_size, out_size, scale, exact, rank), None)]
    elif mode == "bicubic":
        taps = _cubic_taps(in_size, out_size, align_corners, scale)
    else:
        taps = _linear_taps(in_size, out_size, align_corners, scale)
    weighted = taps[0][1] is not None
    dense = weighted and in_size <= _DENSE_PER_TAP * len(taps)
    return _AxisPlan(
        taps=tuple(
            (tuple(indices), None if weights is None else tuple(weights))
            for indices, weights in taps
        ),
        identity=_is_identity(taps, in_size),
        dense=_dense_entries(taps, in_size) if dense else None,
    )


def resample(
    x: Tensor,
    size: int | Sequence[int] | None,
    scale_factor: float | Sequence[float] | None,
    mode: str,
    align_corners: bool | None,
    recompute_scale_factor: bool | None,
    antialias: bool,
) -> Tensor:
    """Resample ``x`` as :func:`~lucid.nn.functional.interpolate` documents.

    Checks the mode against the input, resolves the output size and the
    scale each axis maps with, then runs the engine kernel when it lands
    every sample where the reference does, and the per-axis gathers when
    it does not.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, *spatial)``.
    size : int or sequence of int, optional
        Output spatial size.
    scale_factor : float or sequence of float, optional
        Per-axis factor; the output size is ``floor(in * factor)``.
    mode : str
        Resampling mode.
    align_corners : bool, optional
        Corner convention for the interpolating modes.
    recompute_scale_factor : bool, optional
        Map by ``in / out`` rather than ``1 / scale_factor``.
    antialias : bool
        Widen the bilinear or bicubic filter when downsampling.

    Returns
    -------
    Tensor
        ``x`` with its spatial axes resampled.
    """
    _check_mode(x, mode, align_corners, antialias)
    sizes, scales = _resolve(x, size, scale_factor, recompute_scale_factor)
    spatial = [int(s) for s in x.shape[2:]]
    corners = bool(align_corners)

    if mode == "area":
        # The reference's area mode is adaptive average pooling, scale or not.
        if len(sizes) == 1:
            return adaptive_avg_pool1d(x, sizes[0])
        if len(sizes) == 2:
            return adaptive_avg_pool2d(x, (sizes[0], sizes[1]))
        return adaptive_avg_pool3d(x, (sizes[0], sizes[1], sizes[2]))
    kernel = (mode == "nearest" or mode in _LINEAR) and not antialias
    if (
        kernel
        and x.is_floating_point()
        and _engine_maps_like(mode, spatial, sizes, scales, corners)
    ):
        return _engine_resample(x, mode, sizes, corners)

    out = x
    # Last axis first: it multiplies without a copy, and a shrinking image
    # makes every later product smaller.
    for axis in reversed(range(2, x.ndim)):
        n, m, s = spatial[axis - 2], sizes[axis - 2], scales[axis - 2]
        # Only plain nearest depends on the rank; leaving it out of the
        # other modes' key lets every rank share their plans.
        rank = len(spatial) if mode == "nearest" else 0
        plan = _axis_plan(mode, n, m, corners, s, antialias, rank)
        if plan.identity:
            continue
        if plan.dense is not None:
            out = _apply_dense(out, axis, plan)
        else:
            out = _apply_gathers(out, axis, plan.taps)
    return out.clone() if out is x else out
