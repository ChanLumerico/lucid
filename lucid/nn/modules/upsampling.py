"""
Upsampling and pixel-shuffle modules.
"""

from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.sampling import interpolate


class Upsample(Module):
    """Upsample input to a given size or scale factor.

    Args:
        size:          Output spatial size (H, W) or single int.
        scale_factor:  Multiplier for spatial size.
        mode:          Interpolation algorithm ('nearest', 'bilinear', 'trilinear').
        align_corners: If True, align input and output corners.
    """

    def __init__(
        self,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        return interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

    def extra_repr(self) -> str:
        parts = []
        if self.size is not None:
            parts.append(f"size={self.size}")
        if self.scale_factor is not None:
            parts.append(f"scale_factor={self.scale_factor}")
        parts.append(f"mode={self.mode!r}")
        return ", ".join(parts)


class PixelShuffle(Module):
    """Rearrange (N, C*r^2, H, W) → (N, C, H*r, W*r).

    Args:
        upscale_factor: Upscaling factor r.
    """

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        r = self.upscale_factor
        n, c_r2, h, w = x.shape
        c = c_r2 // (r * r)
        impl = _unwrap(x)
        t = _C_engine.reshape(impl, [n, c, r, r, h, w])
        t = _C_engine.permute(t, [0, 1, 4, 2, 5, 3])
        return _wrap(_C_engine.reshape(t, [n, c, h * r, w * r]))

    def extra_repr(self) -> str:
        return f"upscale_factor={self.upscale_factor}"


class PixelUnshuffle(Module):
    """Inverse of PixelShuffle: (N, C, H*r, W*r) → (N, C*r^2, H, W).

    Args:
        downscale_factor: Downscaling factor r.
    """

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: Tensor) -> Tensor:
        r = self.downscale_factor
        n, c, h_r, w_r = x.shape
        h, w = h_r // r, w_r // r
        impl = _unwrap(x)
        t = _C_engine.reshape(impl, [n, c, h, r, w, r])
        t = _C_engine.permute(t, [0, 1, 3, 5, 2, 4])
        return _wrap(_C_engine.reshape(t, [n, c * r * r, h, w]))

    def extra_repr(self) -> str:
        return f"downscale_factor={self.downscale_factor}"
