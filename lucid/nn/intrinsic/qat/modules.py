"""QAT ``ConvBnReLU2d`` — BatchNorm folded into the conv during training.

The delicate QAT primitive: it keeps a trainable conv + BatchNorm, but on
every forward folds BN's (running) affine into the conv weight, fake-quantizes
the **folded** weight, convolves, applies ReLU, and fake-quantizes the output.
Gradients flow (via STE) to the conv weight and BN parameters, so the network
learns weights that survive the eventual folded-and-quantized inference.
:func:`convert` bakes the folded int8 weight into a quantized
:class:`~lucid.nn.quantized.ConvReLU2d`.
"""

from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig


class ConvBnReLU2d(nn.Module):
    """Fused Conv+BN(+ReLU) with fold-and-fake-quant, trainable via STE."""

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        relu: bool = True,
        qconfig: QConfig | None = None,
    ) -> None:
        super().__init__()
        if qconfig is None:
            raise ValueError("qat.ConvBnReLU2d requires a qconfig")
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    def _fold(self) -> tuple[Tensor, Tensor]:
        """Return the BN-folded ``(weight, bias)`` for the conv."""
        bn, conv = self.bn, self.conv
        running_var = cast("Tensor", bn.running_var)
        running_mean = cast("Tensor", bn.running_mean)
        inv_std = (running_var + bn.eps).rsqrt()
        gamma = cast("Tensor", bn.weight) if bn.affine else lucid.ones_like(inv_std)
        scale = gamma * inv_std
        out_channels = conv.weight.shape[0]
        w = conv.weight * scale.reshape((out_channels, 1, 1, 1))
        conv_bias = conv.bias if conv.bias is not None else lucid.zeros_like(scale)
        bias = (conv_bias - running_mean) * scale
        if bn.affine:
            bias = bias + cast("Tensor", bn.bias)
        return w, bias

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary layer
        """Fold BN → fake-quant folded weight → conv → ReLU → fake-quant output."""
        weight, bias = self._fold()
        w_q = cast("Tensor", self.weight_fake_quant(weight))
        conv = self.conv
        y = F.conv2d(
            x, w_q, bias, conv.stride, conv.padding, conv.dilation, conv.groups
        )
        if self.relu:
            y = F.relu(y)
        return cast("Tensor", self.activation_post_process(y))


def convbnrelu2d_to_quantized(mod: nn.Module) -> nn.Module:
    """Bake a trained :class:`ConvBnReLU2d` into a quantized inference conv."""
    from lucid.nn.quantized.conv import Conv2d as QConv2d
    from lucid.nn.quantized.intrinsic import ConvReLU2d as QConvReLU2d
    from lucid.quantization._functional import quantize

    cbr = cast("ConvBnReLU2d", mod)
    weight, bias = cbr._fold()
    wfq = cbr.weight_fake_quant
    wfq(weight)  # observe the folded weight
    w_scale, w_zp = wfq.calculate_qparams()
    ch_axis = wfq.ch_axis if wfq.ch_axis is not None else 0
    codes = quantize(weight, w_scale, w_zp, wfq.qdtype, ch_axis=wfq.ch_axis)

    conv = cbr.conv
    q_cls = QConvReLU2d if cbr.relu else QConv2d
    q = q_cls(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
    )
    q.register_buffer("weight_int8", codes)
    q.register_buffer("weight_scale", w_scale)
    q.register_buffer("weight_zero_point", w_zp)
    q.weight_ch_axis = ch_axis
    q.register_buffer("bias", bias)
    a_scale, a_zp = cbr.activation_post_process.calculate_qparams()
    q.register_buffer("scale", a_scale)
    q.register_buffer("zero_point", a_zp)
    q.out_qdtype = cbr.activation_post_process.qdtype
    return q
