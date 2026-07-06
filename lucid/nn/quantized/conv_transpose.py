"""Quantized ``ConvTranspose{1,2,3}d`` — int8 weight, dequant-to-float forward.

Same sidecar design (B) as the quantized :class:`~lucid.nn.quantized.Conv2d`
family: the transposed-conv kernel is stored int8 (**per-tensor** symmetric —
the transposed weight layout ``(in, out/groups, *k)`` makes a per-output-channel
axis ambiguous, so per-tensor matches the reference default), the forward
dequantizes it, runs the ordinary transposed convolution, then fake-quantizes
the output to the calibrated activation grid.
"""

from typing import TYPE_CHECKING, Protocol, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import activation_qparams
from lucid.quantization._functional import dequantize, fake_quantize, quantize
from lucid.quantization._qscheme import QDtype, per_channel_symmetric, qint8, quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    _IntTuple = int | tuple[int, ...]

    class _FloatConvT(Protocol):
        """Structural view of a calibrated float transposed conv."""

        in_channels: int
        out_channels: int
        kernel_size: _IntTuple
        stride: _IntTuple
        padding: _IntTuple | str
        output_padding: _IntTuple
        dilation: _IntTuple
        groups: int
        weight: Tensor
        bias: Tensor | None
        qconfig: object


class _QuantizedConvTransposeNd(nn.Module):
    """Shared implementation for the quantized transposed-conv family."""

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor
    scale: Tensor
    zero_point: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: "_IntTuple",
        stride: "_IntTuple",
        padding: "_IntTuple",
        output_padding: "_IntTuple",
        dilation: "_IntTuple",
        groups: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.out_qdtype: QDtype = quint8
        # Per-output-channel weight quant on the transposed-weight axis 1
        # (``(in, out/groups, *k)``) — matches the reference default and is much
        # tighter than per-tensor for wide-range (esp. 3d) kernels.
        self.weight_ch_axis = 1
        n_out = out_channels // groups
        ks = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        weight_shape = (in_channels, n_out, *ks)
        self.register_buffer("weight_int8", lucid.zeros(weight_shape, dtype=lucid.int8))
        self.register_buffer("weight_scale", lucid.ones(n_out))
        self.register_buffer("weight_zero_point", lucid.zeros(n_out))
        if bias:
            self.register_buffer("bias", lucid.zeros(out_channels))
        else:
            self.bias = None
        self.register_buffer("scale", lucid.tensor(1.0))
        self.register_buffer("zero_point", lucid.tensor(0.0))

    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        """Run the rank-specific transposed convolution — overridden per subclass."""
        raise NotImplementedError

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary layer
        """Dequantize the kernel, transposed-convolve, fake-quantize the output."""
        weight = dequantize(
            self.weight_int8,
            self.weight_scale,
            self.weight_zero_point,
            ch_axis=self.weight_ch_axis,
        )
        y = self._conv_forward(x, weight)
        return fake_quantize(
            y,
            self.scale,
            self.zero_point,
            self.out_qdtype.quant_min,
            self.out_qdtype.quant_max,
        )

    @classmethod
    def from_float(cls, mod: nn.Module) -> "_QuantizedConvTransposeNd":
        """Quantize a calibrated float transposed conv (per-channel int8 weight)."""
        from lucid.quantization.observer import PerChannelMinMaxObserver

        f = cast("_FloatConvT", mod)
        if isinstance(f.padding, str):
            raise NotImplementedError(
                "quantized conv_transpose: string padding is not supported yet"
            )
        qmod = cls(
            f.in_channels,
            f.out_channels,
            f.kernel_size,
            f.stride,
            f.padding,
            f.output_padding,
            f.dilation,
            f.groups,
            bias=f.bias is not None,
        )
        wobs = PerChannelMinMaxObserver(
            ch_axis=1, qscheme=per_channel_symmetric, qdtype=qint8
        )
        wobs(f.weight)
        w_scale, w_zp = wobs.calculate_qparams()
        qmod.register_buffer(
            "weight_int8", quantize(f.weight, w_scale, w_zp, qint8, ch_axis=1)
        )
        qmod.register_buffer("weight_scale", w_scale)
        qmod.register_buffer("weight_zero_point", w_zp)
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


class ConvTranspose1d(_QuantizedConvTransposeNd):
    """Quantized 1-D transposed convolution."""

    @override
    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.conv_transpose1d(
            x,
            weight,
            self.bias,
            cast("tuple[int]", self.stride),
            cast("tuple[int]", self.padding),
            cast("tuple[int]", self.output_padding),
            self.groups,
            cast("tuple[int]", self.dilation),
        )


class ConvTranspose2d(_QuantizedConvTransposeNd):
    """Quantized 2-D transposed convolution."""

    @override
    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.conv_transpose2d(
            x,
            weight,
            self.bias,
            cast("tuple[int, int]", self.stride),
            cast("tuple[int, int]", self.padding),
            cast("tuple[int, int]", self.output_padding),
            self.groups,
            cast("tuple[int, int]", self.dilation),
        )


class ConvTranspose3d(_QuantizedConvTransposeNd):
    """Quantized 3-D transposed convolution."""

    @override
    def _conv_forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.conv_transpose3d(
            x,
            weight,
            self.bias,
            cast("tuple[int, int, int]", self.stride),
            cast("tuple[int, int, int]", self.padding),
            cast("tuple[int, int, int]", self.output_padding),
            self.groups,
            cast("tuple[int, int, int]", self.dilation),
        )
