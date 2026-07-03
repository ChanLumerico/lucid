"""Quantized ``Conv1d`` / ``Conv2d`` / ``Conv3d`` — int8 weights, float compute.

Same sidecar recipe as the quantized :class:`~lucid.nn.quantized.Linear`:
the kernel is stored as int8 codes with per-output-channel ``scale`` /
``zero_point``; the forward dequantizes it, runs the ordinary convolution,
and fake-quantizes the output to the calibrated activation grid.  Integer /
tuple padding with ``padding_mode="zeros"`` is supported (covers the vision
zoo); string ``"same"`` / ``"valid"`` padding is deferred.
"""

from typing import TYPE_CHECKING, Protocol, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import activation_qparams, quantize_weight
from lucid.quantization._functional import dequantize, fake_quantize
from lucid.quantization._qscheme import QDtype, quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    _IntTuple = tuple[int, ...]

    class _FloatConv(Protocol):
        """Structural view of a calibrated float conv module."""

        in_channels: int
        out_channels: int
        kernel_size: _IntTuple
        stride: _IntTuple
        padding: _IntTuple | str
        dilation: _IntTuple
        groups: int
        weight: Tensor
        bias: Tensor | None
        qconfig: object


class _QuantizedConvNd(nn.Module):
    """Shared implementation for the quantized convolution family."""

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor
    scale: Tensor
    zero_point: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _IntTuple,
        stride: _IntTuple,
        padding: _IntTuple,
        dilation: _IntTuple,
        groups: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight_ch_axis = 0
        self.out_qdtype: QDtype = quint8
        weight_shape = (out_channels, in_channels // groups, *kernel_size)
        self.register_buffer("weight_int8", lucid.zeros(weight_shape, dtype=lucid.int8))
        self.register_buffer("weight_scale", lucid.ones(out_channels))
        self.register_buffer("weight_zero_point", lucid.zeros(out_channels))
        if bias:
            self.register_buffer("bias", lucid.zeros(out_channels))
        else:
            self.bias = None
        self.register_buffer("scale", lucid.tensor(1.0))
        self.register_buffer("zero_point", lucid.tensor(0.0))

    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        """Run the rank-specific convolution — overridden per subclass."""
        raise NotImplementedError

    def _activation(self, y: Tensor) -> Tensor:
        """Post-conv activation hook (identity; ReLU in the fused variant)."""
        return y

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary layer
        """Dequantize the kernel, convolve, fake-quantize the output."""
        weight = dequantize(
            self.weight_int8,
            self.weight_scale,
            self.weight_zero_point,
            ch_axis=self.weight_ch_axis,
        )
        y = self._activation(self._conv_forward(x, weight))
        return fake_quantize(
            y,
            self.scale,
            self.zero_point,
            self.out_qdtype.quant_min,
            self.out_qdtype.quant_max,
        )

    @classmethod
    def from_float(cls, mod: nn.Module) -> _QuantizedConvNd:
        """Quantize a calibrated float convolution module."""
        f = cast("_FloatConv", mod)
        if isinstance(f.padding, str):
            raise NotImplementedError(
                "quantized conv: string padding ('same'/'valid') is not supported yet"
            )
        has_bias = f.bias is not None
        qmod = cls(
            f.in_channels,
            f.out_channels,
            f.kernel_size,
            f.stride,
            f.padding,
            f.dilation,
            f.groups,
            bias=has_bias,
        )
        codes, w_scale, w_zp, ch_axis = quantize_weight(mod)
        qmod.register_buffer("weight_int8", codes)
        qmod.register_buffer("weight_scale", w_scale)
        qmod.register_buffer("weight_zero_point", w_zp)
        qmod.weight_ch_axis = ch_axis
        if f.bias is not None:
            qmod.register_buffer("bias", f.bias.detach())

        a_scale, a_zp, a_qdtype = activation_qparams(mod)
        qmod.register_buffer("scale", a_scale)
        qmod.register_buffer("zero_point", a_zp)
        qmod.out_qdtype = a_qdtype
        return qmod

    @override
    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, qdtype={self.out_qdtype.name}"
        )


class Conv1d(_QuantizedConvNd):
    """Quantized 1-D convolution."""

    @override
    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.conv1d(
            x,
            weight,
            self.bias,
            cast("tuple[int]", self.stride),
            cast("tuple[int]", self.padding),
            cast("tuple[int]", self.dilation),
            self.groups,
        )


class Conv2d(_QuantizedConvNd):
    """Quantized 2-D convolution."""

    @override
    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.conv2d(
            x,
            weight,
            self.bias,
            cast("tuple[int, int]", self.stride),
            cast("tuple[int, int]", self.padding),
            cast("tuple[int, int]", self.dilation),
            self.groups,
        )


class Conv3d(_QuantizedConvNd):
    """Quantized 3-D convolution."""

    @override
    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.conv3d(
            x,
            weight,
            self.bias,
            cast("tuple[int, int, int]", self.stride),
            cast("tuple[int, int, int]", self.padding),
            cast("tuple[int, int, int]", self.dilation),
            self.groups,
        )
