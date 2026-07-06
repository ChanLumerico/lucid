"""QAT fused Conv+BN(+ReLU) — BatchNorm folded into the conv during training.

The delicate QAT primitive: a trainable conv + BatchNorm that, on every forward,
folds BN's (running) affine into the conv weight, fake-quantizes the **folded**
weight, convolves, optionally applies ReLU, and fake-quantizes the output.
Gradients flow (via STE) to the conv weight and BN parameters, so the network
learns weights that survive the eventual folded-and-quantized inference.
:func:`convert` bakes the folded int8 weight into a quantized conv.

The fold is rank-generic (1d / 2d / 3d): the only rank-specific piece is the
per-output-channel reshape and the ``F.convNd`` call, both derived from the
conv weight's rank.
"""

from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig

_CONV_FNS = (F.conv1d, F.conv2d, F.conv3d)


class _ConvBnNd(nn.Module):
    """Rank-generic fused Conv+BN(+ReLU) with fold-and-fake-quant (trainable)."""

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(
        self,
        conv: nn.Module,
        bn: nn.Module,
        relu: bool,
        qconfig: QConfig | None,
    ) -> None:
        super().__init__()
        if qconfig is None:
            raise ValueError(f"{type(self).__name__} requires a qconfig")
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    def _fold(self) -> tuple[Tensor, Tensor]:
        """Return the BN-folded ``(weight, bias)`` for the conv (any rank)."""
        conv = cast("nn.Conv2d", self.conv)  # structural: all conv ranks share these
        bn = cast("nn.BatchNorm2d", self.bn)
        running_var = cast("Tensor", bn.running_var)
        running_mean = cast("Tensor", bn.running_mean)
        inv_std = (running_var + bn.eps).rsqrt()
        gamma = cast("Tensor", bn.weight) if bn.affine else lucid.ones_like(inv_std)
        scale = gamma * inv_std
        out_channels = conv.weight.shape[0]
        # (C, 1, …) with one trailing 1 per non-output weight axis → 1d/2d/3d.
        w = conv.weight * scale.reshape((out_channels,) + (1,) * (len(conv.weight.shape) - 1))
        conv_bias = conv.bias if conv.bias is not None else lucid.zeros_like(scale)
        bias = (conv_bias - running_mean) * scale
        if bn.affine:
            bias = bias + cast("Tensor", bn.bias)
        return w, bias

    def _conv(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """Dispatch to ``F.conv{1,2,3}d`` by the conv weight's rank."""
        conv = cast("nn.Conv2d", self.conv)
        fn = _CONV_FNS[len(conv.weight.shape) - 3]
        # ``conv`` is cast to Conv2d structurally; at runtime its rank matches the
        # selected ``fn``, so the (int|tuple) stride/padding are valid for it.
        return fn(x, w, b, conv.stride, conv.padding, conv.dilation, conv.groups)  # type: ignore[arg-type]

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary layer
        """Fold BN → fake-quant folded weight → conv → (ReLU) → fake-quant output."""
        weight, bias = self._fold()
        w_q = cast("Tensor", self.weight_fake_quant(weight))
        y = self._conv(x, w_q, bias)
        if self.relu:
            y = F.relu(y)
        return cast("Tensor", self.activation_post_process(y))


class ConvBn1d(_ConvBnNd):
    """QAT fused Conv1d + BatchNorm1d (no ReLU)."""

    def __init__(self, conv: nn.Module, bn: nn.Module, qconfig: QConfig | None = None) -> None:
        super().__init__(conv, bn, False, qconfig)


class ConvBn2d(_ConvBnNd):
    """QAT fused Conv2d + BatchNorm2d (no ReLU)."""

    def __init__(self, conv: nn.Module, bn: nn.Module, qconfig: QConfig | None = None) -> None:
        super().__init__(conv, bn, False, qconfig)


class ConvBn3d(_ConvBnNd):
    """QAT fused Conv3d + BatchNorm3d (no ReLU)."""

    def __init__(self, conv: nn.Module, bn: nn.Module, qconfig: QConfig | None = None) -> None:
        super().__init__(conv, bn, False, qconfig)


class ConvBnReLU1d(_ConvBnNd):
    """QAT fused Conv1d + BatchNorm1d + ReLU."""

    def __init__(
        self, conv: nn.Module, bn: nn.Module, relu: bool = True, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(conv, bn, relu, qconfig)


class ConvBnReLU2d(_ConvBnNd):
    """QAT fused Conv2d + BatchNorm2d + ReLU."""

    def __init__(
        self, conv: nn.Module, bn: nn.Module, relu: bool = True, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(conv, bn, relu, qconfig)


class ConvBnReLU3d(_ConvBnNd):
    """QAT fused Conv3d + BatchNorm3d + ReLU."""

    def __init__(
        self, conv: nn.Module, bn: nn.Module, relu: bool = True, qconfig: QConfig | None = None
    ) -> None:
        super().__init__(conv, bn, relu, qconfig)


def _convbn_to_quantized(mod: nn.Module) -> nn.Module:
    """Bake a trained fused Conv+BN(+ReLU) into a quantized inference conv (any rank)."""
    import lucid.nn.quantized as nnq
    from lucid.quantization._functional import quantize

    cbr = cast("_ConvBnNd", mod)
    weight, bias = cbr._fold()
    wfq = cbr.weight_fake_quant
    wfq(weight)  # observe the folded weight
    w_scale, w_zp = wfq.calculate_qparams()
    ch_axis = wfq.ch_axis if wfq.ch_axis is not None else 0
    codes = quantize(weight, w_scale, w_zp, wfq.qdtype, ch_axis=wfq.ch_axis)

    conv = cast("nn.Conv2d", cbr.conv)
    rank = len(conv.weight.shape) - 2  # 1 / 2 / 3
    plain = (nnq.Conv1d, nnq.Conv2d, nnq.Conv3d)[rank - 1]
    fused = (nnq.ConvReLU1d, nnq.ConvReLU2d, nnq.ConvReLU3d)[rank - 1]
    q_cls = fused if cbr.relu else plain
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


# Back-compat alias — the original 2d-only name (still imported by convert).
convbnrelu2d_to_quantized = _convbn_to_quantized
