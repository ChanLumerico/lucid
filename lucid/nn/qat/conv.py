"""QAT ``Conv1d`` / ``Conv2d`` / ``Conv3d`` — weight + output fake-quant.

Same idea as the QAT :class:`~lucid.nn.qat.Linear`: a trainable float kernel
with fake-quant on the weight and the output, so gradients (via STE) adapt
the weights to the eventual int8 grid.  Integer / tuple padding is supported
(string ``"same"`` / ``"valid"`` is deferred, as in the quantized conv).
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig


def _conv_from_float(cls: type, mod: nn.Module) -> nn.Module:
    """Build a QAT conv from a float conv, sharing its trained kernel."""
    c = cast("nn.Conv2d", mod)  # structural: all conv ranks share these attrs
    qat = cast(
        "nn.Conv2d",
        cls(
            c.in_channels,
            c.out_channels,
            c.kernel_size,
            stride=c.stride,
            padding=c.padding,
            dilation=c.dilation,
            groups=c.groups,
            bias=c.bias is not None,
            qconfig=cast("QConfig", mod.qconfig),  # set by prepare_qat
        ),
    )
    # Adopt the trained kernel/bias directly — the prepare_qat deep-copy
    # already produced independent Parameters for this module tree.
    qat.weight = c.weight
    if c.bias is not None:
        qat.bias = c.bias
    return qat


class Conv1d(nn.Conv1d):
    """Quantization-aware 1-D convolution."""

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(
        self, *args: object, qconfig: QConfig | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # forwarded to nn.Conv1d
        if qconfig is None:
            raise ValueError("qat conv requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary conv layer
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.conv1d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return cast("Tensor", self.activation_post_process(y))

    @classmethod
    def from_float(cls, mod: nn.Module) -> Conv1d:
        return cast("Conv1d", _conv_from_float(cls, mod))


class Conv2d(nn.Conv2d):
    """Quantization-aware 2-D convolution."""

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(
        self, *args: object, qconfig: QConfig | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # forwarded to nn.Conv2d
        if qconfig is None:
            raise ValueError("qat conv requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary conv layer
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.conv2d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return cast("Tensor", self.activation_post_process(y))

    @classmethod
    def from_float(cls, mod: nn.Module) -> Conv2d:
        return cast("Conv2d", _conv_from_float(cls, mod))


class Conv3d(nn.Conv3d):
    """Quantization-aware 3-D convolution."""

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(
        self, *args: object, qconfig: QConfig | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # forwarded to nn.Conv3d
        if qconfig is None:
            raise ValueError("qat conv requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary conv layer
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.conv3d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return cast("Tensor", self.activation_post_process(y))

    @classmethod
    def from_float(cls, mod: nn.Module) -> Conv3d:
        return cast("Conv3d", _conv_from_float(cls, mod))
