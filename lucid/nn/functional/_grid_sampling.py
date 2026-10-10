"""
Grid sampling the engine kernel does not cover, assembled from ops.

The engine's ``grid_sample`` reads a 4-D input with a bilinear or nearest
filter and pads with zeros or the border; its ``affine_grid`` builds 2-D
grids.  Everything else :func:`~lucid.nn.functional.grid_sample` and
:func:`~lucid.nn.functional.affine_grid` accept is built here:

- **reflection padding**, as a reflection of the grid itself, so the
  4-D case still runs on the kernel (with border padding after it);
- **5-D (volumetric) input**, trilinear or nearest;
- **bicubic**, sixteen taps of the cubic convolution kernel;
- **3-D affine grids**.

Each follows the reference framework's coordinate rules step by step —
unnormalise, reflect, clip, then filter — so a sample lands on the same
taps with the same weights.  The composites are differentiable in both
the input (through ``gather``) and the grid (through the filter weights),
the same two gradients the kernel provides.
"""

import itertools
import math
from typing import TYPE_CHECKING

import lucid as _lucid

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


_BICUBIC_A = -0.75


def cubic_coefficients[T: (float, Tensor)](t: T) -> tuple[T, T, T, T]:
    """Weights of the four taps a bicubic sample reads, at offset ``t``.

    The cubic convolution kernel with :math:`A = -0.75`, which is the
    coefficient the reference implementations use.  The weights are not
    normalised — bicubic is allowed to overshoot, and clipping it here
    would make it a different filter.  ``t`` may be a number (one sample)
    or a tensor (one weight per sample); the polynomial is the same.

    Parameters
    ----------
    t : float or Tensor
        Fractional distance from the second of the four taps, in ``[0, 1)``.

    Returns
    -------
    tuple of four float or Tensor
        Weights for taps at ``-1, 0, +1, +2`` relative to ``floor``.
    """
    a = _BICUBIC_A

    def near(x: T) -> T:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0

    def far(x: T) -> T:
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a

    other = 1.0 - t
    return far(t + 1.0), near(t), near(other), far(other + 1.0)


def reflect(coord: Tensor, low: float, span: float) -> Tensor:
    """Fold ``coord`` back into ``[low, low + span]`` by mirroring at both ends.

    The reference's ``reflect_coordinates``: the distance from ``low``
    is split into whole spans (``flips``) and a remainder, and an odd
    number of flips reads the remainder from the far end.  The result is
    piecewise linear in ``coord`` with slope ``±1``, which is the grid
    gradient reflection padding has.

    Parameters
    ----------
    coord : Tensor
        Coordinates, any shape.
    low : float
        Lower mirror.
    span : float
        Distance to the upper mirror; positive.

    Returns
    -------
    Tensor
        ``coord`` reflected into the interval, same shape.
    """
    distance = (coord - low).abs()
    flips = (distance / span).floor()
    extra = _lucid.fmod(distance, span)
    odd = _lucid.fmod(flips, 2.0) == 1.0
    return _lucid.where(odd, span - extra, extra) + low


def _unnormalize(coord: Tensor, size: int, align_corners: bool) -> Tensor:
    """Map a normalised ``[-1, 1]`` coordinate onto pixel indices of an axis.

    Parameters
    ----------
    coord : Tensor
        Normalised coordinates.
    size : int
        Length of the axis sampled.
    align_corners : bool
        ``True``: ``±1`` are the centres of the end pixels; ``False``:
        their outer edges.

    Returns
    -------
    Tensor
        Pixel-space coordinates, same shape.
    """
    if align_corners:
        return ((coord + 1.0) / 2.0) * (size - 1)
    return ((coord + 1.0) * size - 1.0) / 2.0


def _pad(coord: Tensor, size: int, padding_mode: str, align_corners: bool) -> Tensor:
    """Bring a pixel-space coordinate inside the axis as ``padding_mode`` says.

    ``zeros`` leaves it where it is (the read is masked instead),
    ``border`` clips it, and ``reflection`` mirrors it about the image's
    edges — the end pixels' centres with ``align_corners``, their outer
    edges without — and then clips, as the reference does.

    Parameters
    ----------
    coord : Tensor
        Pixel-space coordinates.
    size : int
        Length of the axis.
    padding_mode : str
        ``"zeros"``, ``"border"`` or ``"reflection"``.
    align_corners : bool
        Corner convention the coordinate was unnormalised with.

    Returns
    -------
    Tensor
        The padded coordinates, same shape.
    """
    if padding_mode == "zeros":
        return coord
    if padding_mode == "reflection":
        if not align_corners:
            coord = reflect(coord, -0.5, float(size))
        elif size > 1:
            coord = reflect(coord, 0.0, float(size - 1))
        else:
            # One pixel, mirrored at its own centre: every read is it.
            coord = _lucid.zeros_like(coord)
    return coord.clip(0.0, float(size - 1))


def _gather(flat: Tensor, coords: list[Tensor], sizes: list[int]) -> Tensor:
    """Read ``flat`` at integral pixel coordinates; reads off the image are 0.

    Parameters
    ----------
    flat : Tensor
        Input with its spatial axes flattened, ``(N, C, S)``.
    coords : list of Tensor
        One ``(N, P)`` tensor of integral coordinates per spatial axis,
        fastest-varying (``x``) first.
    sizes : list of int
        The matching axis lengths.

    Returns
    -------
    Tensor
        ``(N, C, P)`` samples, zero where any coordinate is off its axis.
    """
    n, c, _ = flat.shape
    index = _lucid.zeros_like(coords[0], dtype=_lucid.int64)
    inside = _lucid.ones_like(coords[0], dtype=_lucid.bool)
    stride = 1
    for coord, size in zip(coords, sizes, strict=True):
        last = float(size - 1)
        inside = inside & (coord >= 0.0) & (coord <= last)
        index = index + coord.clip(0.0, last).to(_lucid.int64) * stride
        stride *= size
    points = int(index.shape[1])
    values = _lucid.gather(flat, 2, index.unsqueeze(1).expand(n, c, points))
    return values * inside.unsqueeze(1).to(flat.dtype)


def _total(terms: list[Tensor]) -> Tensor:
    """Add up per-tap contributions, in tap order.

    Parameters
    ----------
    terms : list of Tensor
        Non-empty list of same-shape tensors.

    Returns
    -------
    Tensor
        Their sum.
    """
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _linear(flat: Tensor, coords: list[Tensor], sizes: list[int]) -> Tensor:
    """Bilinear / trilinear: the ``2**k`` surrounding pixels, weighted.

    Parameters
    ----------
    flat : Tensor
        ``(N, C, S)`` input.
    coords : list of Tensor
        Padded pixel-space coordinates, one ``(N, P)`` per axis, ``x`` first.
    sizes : list of int
        Axis lengths.

    Returns
    -------
    Tensor
        ``(N, C, P)`` samples.
    """
    lower = [coord.floor() for coord in coords]
    frac = [coord - base for coord, base in zip(coords, lower, strict=True)]
    terms = []
    for corner in itertools.product((0, 1), repeat=len(coords)):
        weight = _lucid.ones_like(frac[0])
        taps = []
        for upper, base, f in zip(corner, lower, frac, strict=True):
            weight = weight * (f if upper else 1.0 - f)
            taps.append(base + float(upper))
        terms.append(_gather(flat, taps, sizes) * weight.unsqueeze(1))
    return _total(terms)


def _nearest(flat: Tensor, coords: list[Tensor], sizes: list[int]) -> Tensor:
    """Nearest: the one pixel each coordinate rounds to (half to even).

    The sample does not move with the grid, but it still belongs to the
    grid's graph, as the kernel's nearest path does: the grid gets a zero
    gradient, and a caller differentiating only the grid gets that rather
    than an output that does not require grad.  ``round`` has zero slope
    and the clip pins every value to 0 (infinite ones too, which a
    product with 0 would turn into NaN), so the weight is exactly 1.

    Takes and returns what ``_linear`` does: the ``(N, C, S)`` input,
    padded coordinates (``x`` first) with their axis lengths, and
    ``(N, C, P)`` samples.
    """
    nearest = [coord.round() for coord in coords]
    weight = _total([index.clip(0.0, 0.0) for index in nearest]) + 1.0
    return _gather(flat, nearest, sizes) * weight.unsqueeze(1)


def _bicubic(
    flat: Tensor,
    coords: list[Tensor],
    sizes: list[int],
    padding_mode: str,
    align_corners: bool,
) -> Tensor:
    """Bicubic: a 4 x 4 neighbourhood weighted by the cubic kernel.

    The coordinates are *not* padded first — clipping ``-0.1`` to ``0``
    would move ``floor`` and every tap with it.  Each tap is padded on
    its own instead, as the reference's ``get_value_bounded`` does, and
    the weights come from the unpadded fraction.  The grid gradient
    flows through the weights only; the taps are integral.

    Parameters
    ----------
    flat : Tensor
        ``(N, C, H * W)`` input.
    coords : list of Tensor
        Unpadded pixel-space ``[x, y]``, each ``(N, P)``.
    sizes : list of int
        ``[W, H]``.
    padding_mode : str
        Padding applied to each tap.
    align_corners : bool
        Corner convention, for reflection.

    Returns
    -------
    Tensor
        ``(N, C, P)`` samples.
    """
    taps: list[list[Tensor]] = []
    weights: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for coord, size in zip(coords, sizes, strict=True):
        base = coord.floor()
        weights.append(cubic_coefficients(coord - base))
        taps.append(
            [
                _pad(base + float(offset), size, padding_mode, align_corners)
                for offset in (-1, 0, 1, 2)
            ]
        )
    terms = [
        _gather(flat, [taps[0][i], taps[1][j]], sizes)
        * (weights[0][i] * weights[1][j]).unsqueeze(1)
        for j in range(4)
        for i in range(4)
    ]
    return _total(terms)


_MODES = ("bilinear", "nearest", "bicubic")
_PADDINGS = ("zeros", "border", "reflection")


def check_options(x: Tensor, mode: str, padding_mode: str) -> None:
    """Refuse, by name, a ``grid_sample`` setting it cannot compute.

    Parameters
    ----------
    x : Tensor
        The input; its rank decides which modes apply.
    mode : str
        Requested filter.
    padding_mode : str
        Requested padding.

    Raises
    ------
    ValueError
        Unknown ``mode`` or ``padding_mode``, an input that is not 4-D or
        5-D, or ``"bicubic"`` with a 5-D input (the reference has no
        tricubic filter either).
    """
    if mode not in _MODES:
        raise ValueError(
            f"grid_sample: unsupported mode {mode!r}; expected one of {_MODES}"
        )
    if padding_mode not in _PADDINGS:
        raise ValueError(
            f"grid_sample: unsupported padding_mode {padding_mode!r}; "
            f"expected one of {_PADDINGS}"
        )
    if x.ndim not in (4, 5):
        raise ValueError(
            "grid_sample: expected a 4-D (N, C, H, W) or 5-D (N, C, D, H, W) "
            f"input, got shape {tuple(x.shape)}"
        )
    if x.ndim == 5 and mode == "bicubic":
        raise ValueError("grid_sample: mode 'bicubic' supports only 4-D input")


def _as_real(t: Tensor) -> Tensor:
    """Promote an integral tensor to float32, as the engine kernel does.

    Parameters
    ----------
    t : Tensor
        Input or grid.

    Returns
    -------
    Tensor
        ``t`` if it is already floating point, otherwise a float32 copy.
    """
    return t if t.is_floating_point() else t.to(_lucid.float32)


def _check_operands(x: Tensor, grid: Tensor) -> None:
    """Refuse a grid that does not fit the input, before any read.

    Parameters
    ----------
    x : Tensor
        ``(N, C, *spatial)`` input.
    grid : Tensor
        Expected ``(N, *out_spatial, k)`` with ``k = len(spatial)``.

    Raises
    ------
    ValueError
        Shapes or devices do not match.
    TypeError
        Dtypes do not match.
    """
    rank = x.ndim - 2
    if grid.ndim != x.ndim or int(grid.shape[-1]) != rank:
        raise ValueError(
            f"grid_sample: a {x.ndim}-D input needs a {x.ndim}-D grid whose "
            f"last axis has {rank} coordinates, got grid of shape {tuple(grid.shape)}"
        )
    if int(grid.shape[0]) != int(x.shape[0]):
        raise ValueError(
            f"grid_sample: batch size mismatch, input {tuple(x.shape)} "
            f"vs grid {tuple(grid.shape)}"
        )
    if x.device != grid.device:
        raise ValueError(
            f"grid_sample: input is on {x.device} but grid is on {grid.device}"
        )
    if x.dtype != grid.dtype:
        raise TypeError(
            f"grid_sample: input and grid must have the same dtype, "
            f"got {x.dtype} and {grid.dtype}"
        )


def composite_grid_sample(
    x: Tensor, grid: Tensor, mode: str, padding_mode: str, align_corners: bool
) -> Tensor:
    """Sample a 4-D or 5-D input at ``grid`` without the engine kernel.

    Parameters
    ----------
    x : Tensor
        ``(N, C, H, W)`` or ``(N, C, D, H, W)``.
    grid : Tensor
        ``(N, H_out, W_out, 2)`` or ``(N, D_out, H_out, W_out, 3)``.
    mode : str
        ``"bilinear"``, ``"nearest"`` or (4-D only) ``"bicubic"``.
    padding_mode : str
        ``"zeros"``, ``"border"`` or ``"reflection"``.
    align_corners : bool
        Corner convention.

    Returns
    -------
    Tensor
        ``(N, C, *out_spatial)`` samples.
    """
    x = _as_real(x)
    grid = _as_real(grid)
    _check_operands(x, grid)
    n, c = int(x.shape[0]), int(x.shape[1])
    spatial = [int(s) for s in x.shape[2:]]
    out_spatial = [int(s) for s in grid.shape[1:-1]]
    points = math.prod(out_spatial)
    flat = x.reshape(n, c, math.prod(spatial))
    # Grid components are (x, y[, z]) — fastest axis first, the reverse
    # of the input's (D, H, W).
    sizes = spatial[::-1]
    components = _lucid.unbind(grid.reshape(n, points, len(sizes)), -1)
    coords = [
        _unnormalize(component, size, align_corners)
        for component, size in zip(components, sizes, strict=True)
    ]
    if mode == "bicubic":
        out = _bicubic(flat, coords, sizes, padding_mode, align_corners)
    else:
        coords = [
            _pad(coord, size, padding_mode, align_corners)
            for coord, size in zip(coords, sizes, strict=True)
        ]
        sample = _nearest if mode == "nearest" else _linear
        out = sample(flat, coords, sizes)
    return out.reshape(n, c, *out_spatial)


def _base_axis(steps: int, align_corners: bool, like: Tensor) -> Tensor:
    """Normalised centres of ``steps`` output pixels along one axis.

    Parameters
    ----------
    steps : int
        Axis length.
    align_corners : bool
        ``True`` spans ``[-1, 1]`` end-centre to end-centre; ``False``
        pulls the centres in by half a pixel.
    like : Tensor
        Supplies dtype and device.

    Returns
    -------
    Tensor
        ``(steps,)`` coordinates; a single pixel sits at ``0``.
    """
    if steps <= 1:
        return _lucid.zeros(steps, dtype=like.dtype, device=like.device)
    axis = _lucid.linspace(-1.0, 1.0, steps, dtype=like.dtype, device=like.device)
    return axis if align_corners else axis * ((steps - 1) / steps)


def affine_grid_3d(theta: Tensor, size: list[int], align_corners: bool) -> Tensor:
    """Volumetric affine grid: ``theta @ [x, y, z, 1]`` at every output voxel.

    Parameters
    ----------
    theta : Tensor
        ``(N, 3, 4)`` affine matrices.
    size : list of int
        ``(N, C, D, H, W)`` target size.
    align_corners : bool
        Corner convention, as for :func:`composite_grid_sample`.

    Returns
    -------
    Tensor
        ``(N, D, H, W, 3)`` normalised sampling coordinates.

    Raises
    ------
    ValueError
        ``theta`` is not ``(N, 3, 4)`` for the ``N`` in ``size``.
    """
    n, _, d, h, w = size
    if tuple(int(s) for s in theta.shape) != (n, 3, 4):
        raise ValueError(
            f"affine_grid: a 5-D size needs theta of shape ({n}, 3, 4), "
            f"got {tuple(theta.shape)}"
        )
    theta = _as_real(theta)
    xs = _base_axis(w, align_corners, theta).reshape(1, 1, w).expand(d, h, w)
    ys = _base_axis(h, align_corners, theta).reshape(1, h, 1).expand(d, h, w)
    zs = _base_axis(d, align_corners, theta).reshape(d, 1, 1).expand(d, h, w)
    base = _lucid.stack([xs, ys, zs, _lucid.ones_like(xs)], -1).reshape(1, d * h * w, 4)
    return _lucid.matmul(base, theta.transpose(1, 2)).reshape(n, d, h, w, 3)
