"""Quantized ``Linear`` — int8 weight storage, float-carried activations.

Under the sidecar representation (design B) the weight is stored as int8
codes plus per-channel ``scale`` / ``zero_point`` buffers; the forward
dequantizes the weight to ``float32``, runs the ordinary linear op, then
fake-quantizes the output to the calibrated activation grid.  This yields
the *numerics* of int8 inference (so accuracy matches a real int8 kernel)
while the actual GEMM stays in float — the real low-precision GEMM is
swapped in underneath at Phase 6 without changing this surface.
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

    class _FloatLinear(Protocol):
        """Structural view of a calibrated float linear module."""

        in_features: int
        out_features: int
        weight: Tensor
        bias: Tensor | None
        qconfig: object


class Linear(nn.Module):
    """Quantized linear (fully-connected) layer — int8 weight, float compute.

    Under the sidecar representation (design B) the weight is stored as int8
    codes plus per-output-channel ``scale`` / ``zero_point`` buffers; each
    forward dequantizes the weight to float, runs the ordinary ``F.linear``,
    then fake-quantizes the output to the calibrated activation grid. This
    yields the *numerics* of int8 inference (accuracy matches a real int8
    kernel) while the GEMM itself stays in float. Produced from a calibrated
    float :class:`~lucid.nn.Linear` by :func:`lucid.quantization.convert` /
    :meth:`from_float`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        Whether a (float) bias term is added after the linear map. Defaults to
        ``True``.

    Notes
    -----
    The weight is quantized **per-output-channel on axis 0**; the bias stays
    float. For the real low-precision Metal GEMM (weight-only int4/int8, no
    dequantize hop) see :class:`~lucid.nn.quantized.QuantizedLinearMLX`.
    """

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor
    scale: Tensor
    zero_point: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_ch_axis = 0
        self.out_qdtype: QDtype = quint8
        self.register_buffer(
            "weight_int8", lucid.zeros((out_features, in_features), dtype=lucid.int8)
        )
        self.register_buffer("weight_scale", lucid.ones(out_features))
        self.register_buffer("weight_zero_point", lucid.zeros(out_features))
        if bias:
            self.register_buffer("bias", lucid.zeros(out_features))
        else:
            self.bias = None
        self.register_buffer("scale", lucid.tensor(1.0))
        self.register_buffer("zero_point", lucid.tensor(0.0))

    def _activation(self, y: Tensor) -> Tensor:
        """Post-linear activation hook (identity; ReLU in the fused variant)."""
        return y

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary layer
        """Dequantize the weight, run linear, fake-quantize the output."""
        weight = dequantize(
            self.weight_int8,
            self.weight_scale,
            self.weight_zero_point,
            ch_axis=self.weight_ch_axis,
        )
        y = self._activation(F.linear(x, weight, self.bias))
        return fake_quantize(
            y,
            self.scale,
            self.zero_point,
            self.out_qdtype.quant_min,
            self.out_qdtype.quant_max,
        )

    @classmethod
    def from_float(cls, mod: nn.Module) -> Linear:
        """Quantize a calibrated float :class:`~lucid.nn.Linear`."""
        f = cast("_FloatLinear", mod)
        has_bias = f.bias is not None
        qmod = cls(f.in_features, f.out_features, bias=has_bias)

        codes, w_scale, w_zp, ch_axis = quantize_weight(mod)
        qmod.register_buffer("weight_int8", codes)
        qmod.register_buffer("weight_scale", w_scale)
        qmod.register_buffer("weight_zero_point", w_zp)
        qmod.weight_ch_axis = ch_axis
        if f.bias is not None:
            qmod.register_buffer("bias", f.bias.detach())  # bias stays float

        a_scale, a_zp, a_qdtype = activation_qparams(mod)
        qmod.register_buffer("scale", a_scale)
        qmod.register_buffer("zero_point", a_zp)
        qmod.out_qdtype = a_qdtype
        return qmod

    @override
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"qdtype={self.out_qdtype.name}"
        )
