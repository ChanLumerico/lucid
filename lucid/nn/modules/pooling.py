"""
Pooling modules.
"""

from lucid._tensor.tensor import Tensor
from lucid._types import _Size2d, _Size3d
from lucid.nn.module import Module
from lucid.nn.functional.pooling import (
    max_pool1d,
    max_pool2d,
    max_pool3d,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool2d,
    adaptive_max_pool1d,
    adaptive_max_pool3d,
    max_unpool1d,
    max_unpool2d,
    max_unpool3d,
    fractional_max_pool2d,
    fractional_max_pool3d,
)


class MaxPool1d(Module):
    """1D max pooling."""

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        if return_indices:
            raise NotImplementedError(
                "MaxPool1d: return_indices=True is not supported yet."
            )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        return max_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            False,
            self.ceil_mode,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class MaxPool2d(Module):
    """2D max pooling."""

    def __init__(
        self,
        kernel_size: _Size2d,
        stride: _Size2d | None = None,
        padding: _Size2d = 0,
        dilation: _Size2d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        if return_indices:
            raise NotImplementedError(
                "MaxPool2d: return_indices=True is not supported yet."
            )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        return max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            False,
            self.ceil_mode,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class AvgPool1d(Module):
    """1D average pooling."""

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        return avg_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool2d(Module):
    """2D average pooling."""

    def __init__(
        self,
        kernel_size: _Size2d,
        stride: _Size2d | None = None,
        padding: _Size2d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        return avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveAvgPool1d(Module):
    """Adaptive 1D average pooling."""

    def __init__(self, output_size: int | tuple[int, ...]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_avg_pool1d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveAvgPool2d(Module):
    """Adaptive 2D average pooling."""

    def __init__(self, output_size: _Size2d) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_avg_pool2d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveMaxPool2d(Module):
    """Adaptive 2D max pooling."""

    def __init__(self, output_size: _Size2d, return_indices: bool = False) -> None:
        super().__init__()
        if return_indices:
            raise NotImplementedError(
                "AdaptiveMaxPool2d: return_indices=True is not supported yet."
            )
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_max_pool2d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class MaxPool3d(Module):
    """3D max pooling."""

    def __init__(
        self,
        kernel_size: _Size3d,
        stride: _Size3d | None = None,
        padding: _Size3d = 0,
        dilation: _Size3d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        if return_indices:
            raise NotImplementedError(
                "MaxPool3d: return_indices=True is not supported yet."
            )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        return max_pool3d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


class AvgPool3d(Module):
    """3D average pooling."""

    def __init__(
        self,
        kernel_size: _Size3d,
        stride: _Size3d | None = None,
        padding: _Size3d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        return avg_pool3d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveAvgPool3d(Module):
    """Adaptive 3D average pooling."""

    def __init__(self, output_size: _Size3d) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_avg_pool3d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveMaxPool1d(Module):
    """Adaptive 1D max pooling."""

    def __init__(
        self, output_size: int | tuple[int, ...], return_indices: bool = False
    ) -> None:
        super().__init__()
        if return_indices:
            raise NotImplementedError(
                "AdaptiveMaxPool1d: return_indices=True is not supported yet."
            )
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_max_pool1d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveMaxPool3d(Module):
    """Adaptive 3D max pooling."""

    def __init__(self, output_size: _Size3d, return_indices: bool = False) -> None:
        super().__init__()
        if return_indices:
            raise NotImplementedError(
                "AdaptiveMaxPool3d: return_indices=True is not supported yet."
            )
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_max_pool3d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class LPPool1d(Module):
    """1-D power-average pooling: pool = (sum(|x|^p))^(1/p) per window.

    Uses ``unfold_dim`` + abs + pow + sum + pow engine ops; GPU-compatible.
    """

    def __init__(
        self,
        norm_type: float,
        kernel_size: int,
        stride: int | None = None,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _unwrap, _wrap

        xi = _unwrap(x)  # (N, C, L)
        p = float(self.norm_type)
        k = self.kernel_size
        s = self.stride

        # Unfold last dim → (N, C, L_out, k)
        unfolded = _C_engine.unfold_dim(xi, 2, k, s)
        absv = _C_engine.abs(unfolded)
        powered = _C_engine.pow_scalar(absv, p)
        summed = _C_engine.sum(powered, [3], False)  # (N, C, L_out)
        return _wrap(_C_engine.pow_scalar(summed, 1.0 / p))

    def extra_repr(self) -> str:
        return f"norm_type={self.norm_type}, kernel_size={self.kernel_size}, stride={self.stride}"


class LPPool2d(Module):
    """2-D power-average pooling: pool = (sum(|x|^p))^(1/p) per window.

    Uses two sequential ``unfold_dim`` passes; GPU-compatible.
    """

    def __init__(
        self,
        norm_type: float,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        kh, kw = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.kh, self.kw = kh, kw
        if stride is None:
            self.sh, self.sw = kh, kw
        else:
            self.sh, self.sw = (stride, stride) if isinstance(stride, int) else stride
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _unwrap, _wrap

        xi = _unwrap(x)  # (N, C, H, W)
        p = float(self.norm_type)

        # Unfold H-dim (axis 2) → (N, C, H_out, W, kh)
        uh = _C_engine.unfold_dim(xi, 2, self.kh, self.sh)
        # uh shape: (N, C, H_out, W, kh) — now unfold W-dim (axis 3)
        uw = _C_engine.unfold_dim(uh, 3, self.kw, self.sw)
        # uw shape: (N, C, H_out, W_out, kh, kw)

        absv = _C_engine.abs(uw)
        powered = _C_engine.pow_scalar(absv, p)
        summed = _C_engine.sum(powered, [4, 5], False)  # (N, C, H_out, W_out)
        return _wrap(_C_engine.pow_scalar(summed, 1.0 / p))

    def extra_repr(self) -> str:
        return (
            f"norm_type={self.norm_type}, kernel_size=({self.kh},{self.kw}), "
            f"stride=({self.sh},{self.sw})"
        )


# ── P4 fill: MaxUnpool / FractionalMaxPool ────────────────────────────────


class _MaxUnpoolNd(Module):
    """Shared base — stores kernel/stride/padding, defers to functional call."""

    def __init__(
        self,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


class MaxUnpool1d(_MaxUnpoolNd):
    """Inverse of :class:`MaxPool1d`.  Scatters values back into a sparse
    tensor at the indices saved by the forward pool."""

    def forward(
        self,
        x: Tensor,
        indices: Tensor,
        output_size: tuple[int, ...] | None = None,
    ) -> Tensor:
        return max_unpool1d(
            x,
            indices,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_size=output_size,
        )


class MaxUnpool2d(_MaxUnpoolNd):
    """Inverse of :class:`MaxPool2d`."""

    def forward(
        self,
        x: Tensor,
        indices: Tensor,
        output_size: tuple[int, ...] | None = None,
    ) -> Tensor:
        return max_unpool2d(
            x,
            indices,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_size=output_size,
        )


class MaxUnpool3d(_MaxUnpoolNd):
    """Inverse of :class:`MaxPool3d`."""

    def forward(
        self,
        x: Tensor,
        indices: Tensor,
        output_size: tuple[int, ...] | None = None,
    ) -> Tensor:
        return max_unpool3d(
            x,
            indices,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_size=output_size,
        )


class FractionalMaxPool2d(Module):
    """Fractional max-pooling over a 2-D spatial input.

    Stub matching the reference framework's surface — the engine has no
    return-indices random-pool path yet, so calls raise
    :class:`NotImplementedError`.  Wired here so model code that
    *imports* the module loads cleanly.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        output_size: int | tuple[int, int] | None = None,
        output_ratio: float | tuple[float, float] | None = None,
        return_indices: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices

    def forward(self, x: Tensor) -> Tensor:
        return fractional_max_pool2d(
            x,
            kernel_size=self.kernel_size,
            output_size=self.output_size,
            output_ratio=self.output_ratio,
            return_indices=self.return_indices,
        )


class FractionalMaxPool3d(Module):
    """Fractional max-pooling over a 3-D spatial input — stub (see
    :class:`FractionalMaxPool2d`)."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        output_size: int | tuple[int, int, int] | None = None,
        output_ratio: float | tuple[float, float, float] | None = None,
        return_indices: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices

    def forward(self, x: Tensor) -> Tensor:
        return fractional_max_pool3d(
            x,
            kernel_size=self.kernel_size,
            output_size=self.output_size,
            output_ratio=self.output_ratio,
            return_indices=self.return_indices,
        )
