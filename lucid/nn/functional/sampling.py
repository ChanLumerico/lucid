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
                sf = (
                    (scale_factor, scale_factor)
                    if isinstance(scale_factor, float)
                    else scale_factor
                )
                oh = int(x.shape[2] * sf[0])
                ow = int(x.shape[3] * sf[1])
            return _wrap(_C_engine.nn.interpolate_nearest_2d(_unwrap(x), oh, ow))
        if ndim == 3:
            if size is not None:
                od, oh, ow = (size, size, size) if isinstance(size, int) else size
            else:
                assert scale_factor is not None
                sf = (
                    (scale_factor,) * 3
                    if isinstance(scale_factor, float)
                    else scale_factor
                )
                od = int(x.shape[2] * sf[0])
                oh = int(x.shape[3] * sf[1])
                ow = int(x.shape[4] * sf[2])
            return _wrap(_C_engine.nn.interpolate_nearest_3d(_unwrap(x), od, oh, ow))
    if mode == "bilinear":
        if size is not None:
            oh, ow = (size, size) if isinstance(size, int) else size
        else:
            assert scale_factor is not None
            sf = (
                (scale_factor, scale_factor)
                if isinstance(scale_factor, float)
                else scale_factor
            )
            oh = int(x.shape[2] * sf[0])
            ow = int(x.shape[3] * sf[1])
        ac = align_corners if align_corners is not None else False
        return _wrap(_C_engine.nn.interpolate_bilinear(_unwrap(x), oh, ow, ac))
    if mode == "trilinear":
        if size is not None:
            od, oh, ow = (size, size, size) if isinstance(size, int) else size
        else:
            assert scale_factor is not None
            sf = (
                (scale_factor,) * 3 if isinstance(scale_factor, float) else scale_factor
            )
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


def unfold(
    x: Tensor,
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] = 1,
) -> Tensor:
    """Extract sliding local blocks from a batched 4-D input tensor.

    Args:
        x:           Input of shape (N, C, H, W).
        kernel_size: Size of the sliding blocks.
        dilation:    Stride between elements within a sliding block.
        padding:     Implicit zero padding on both sides.
        stride:      Stride of the sliding blocks.

    Returns:
        Tensor of shape (N, C*kH*kW, L) where L = output locations.
    """

    def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
        return (v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]

    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    ph, pw = _pair(padding)
    sh, sw = _pair(stride)
    return _wrap(
        _C_engine.nn.unfold(_unwrap(x), [kh, kw], [sh, sw], [ph, pw], [dh, dw])
    )


def pad(
    x: Tensor,
    padding: tuple[int, ...],
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor:
    """Pad a tensor.

    padding follows PyTorch convention: flat tuple starting from the LAST dimension.
    For example, (l, r) pads the last dim; (l, r, t, b) pads last two dims.
    Internally converts to per-dimension pairs for the engine.
    """
    impl = _unwrap(x)
    ndim = len(impl.shape)
    n_pad_dims = len(padding) // 2
    # Convert PyTorch flat (last→first) to engine per-dim pairs (first→last)
    pad_pairs: list[tuple[int, int]] = [(0, 0)] * ndim
    for i in range(n_pad_dims):
        dim_idx = ndim - 1 - i
        left = padding[2 * i]
        right = padding[2 * i + 1]
        pad_pairs[dim_idx] = (left, right)
    return _wrap(_C_engine.pad(impl, pad_pairs, value))
