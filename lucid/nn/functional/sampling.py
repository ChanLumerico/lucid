"""
nn.functional sampling / interpolation / padding operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def interpolate(
    x: Tensor,
    size: int | tuple[int, ...] | None = None,
    scale_factor: float | tuple[float, ...] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
) -> Tensor:
    """Interpolate a tensor to a given size or scale factor."""
    if mode in ("nearest", "nearest-exact"):
        ndim = x.ndim - 2
        if ndim == 2:
            if size is not None:
                oh, ow = (size, size) if isinstance(size, int) else size
            else:
                assert scale_factor is not None
                sf = (scale_factor, scale_factor) if isinstance(scale_factor, float) else scale_factor
                oh = int(x.shape[2] * sf[0])
                ow = int(x.shape[3] * sf[1])
            return _wrap(_C_engine.nn.interpolate_nearest_2d(_unwrap(x), oh, ow))
        if ndim == 3:
            if size is not None:
                od, oh, ow = (size, size, size) if isinstance(size, int) else size
            else:
                assert scale_factor is not None
                sf = (scale_factor,) * 3 if isinstance(scale_factor, float) else scale_factor
                od = int(x.shape[2] * sf[0])
                oh = int(x.shape[3] * sf[1])
                ow = int(x.shape[4] * sf[2])
            return _wrap(_C_engine.nn.interpolate_nearest_3d(_unwrap(x), od, oh, ow))
    if mode == "bilinear":
        if size is not None:
            oh, ow = (size, size) if isinstance(size, int) else size
        else:
            assert scale_factor is not None
            sf = (scale_factor, scale_factor) if isinstance(scale_factor, float) else scale_factor
            oh = int(x.shape[2] * sf[0])
            ow = int(x.shape[3] * sf[1])
        ac = align_corners if align_corners is not None else False
        return _wrap(_C_engine.nn.interpolate_bilinear(_unwrap(x), oh, ow, ac))
    if mode == "trilinear":
        if size is not None:
            od, oh, ow = (size, size, size) if isinstance(size, int) else size
        else:
            assert scale_factor is not None
            sf = (scale_factor,) * 3 if isinstance(scale_factor, float) else scale_factor
            od = int(x.shape[2] * sf[0])
            oh = int(x.shape[3] * sf[1])
            ow = int(x.shape[4] * sf[2])
        ac = align_corners if align_corners is not None else False
        return _wrap(_C_engine.nn.interpolate_trilinear(_unwrap(x), od, oh, ow, ac))
    raise ValueError(f"Unsupported interpolation mode: {mode!r}")


def grid_sample(
    x: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool | None = None,
) -> Tensor:
    """Sample x using grid coordinates."""
    ac = align_corners if align_corners is not None else False
    return _wrap(_C_engine.nn.grid_sample(_unwrap(x), _unwrap(grid), ac))


def affine_grid(
    theta: Tensor,
    size: list[int] | tuple[int, ...],
    align_corners: bool | None = None,
) -> Tensor:
    """Generate a sampling grid for affine_grid / grid_sample."""
    ac = align_corners if align_corners is not None else False
    return _wrap(_C_engine.nn.affine_grid(_unwrap(theta), list(size), ac))


def pad(
    x: Tensor,
    padding: tuple[int, ...],
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor:
    """
    Pad a tensor.

    padding is specified as (left, right) for 1D, (left, right, top, bottom) for 2D,
    starting from the last dimension (same convention as PyTorch F.pad).
    """
    _MODE_MAP = {"constant": 0, "reflect": 1, "replicate": 2, "circular": 3}
    mode_int = _MODE_MAP.get(mode, 0)
    return _wrap(_C_engine.pad(_unwrap(x), list(padding), mode_int, value))
