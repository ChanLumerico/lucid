"""QAT ``Linear`` — trains in float with weight + activation fake-quant.

During quantization-aware training the layer keeps a **trainable float
weight** but applies :class:`~lucid.quantization.FakeQuantize` to both the
weight and the output, so the network experiences quantization rounding
(via the straight-through estimator) while learning to compensate for it.
``convert`` later reads the trained weight + the fake-quant observers' final
qparams to build the quantized inference :class:`~lucid.nn.quantized.Linear`.
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig


class Linear(nn.Linear):
    """Quantization-aware ``Linear`` (trainable float weight, fake-quant).

    A trainable float :class:`~lucid.nn.Linear` that fake-quantizes both the weight
    and the output on every forward through a straight-through estimator (STE), letting
    the network experience quantization rounding and learn to compensate while training
    in full precision.  Inserted by :func:`lucid.quantization.prepare_qat`;
    :func:`lucid.quantization.convert` reads the trained weight and the observers' final
    qparams to build an inference :class:`~lucid.nn.quantized.Linear`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default=True
        Whether to add a learnable (float) bias term.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.

    Notes
    -----
    The STE rounds the weight in the forward pass but passes gradients straight through,
    so the float weight stays fully trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qconfig: QConfig | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        if qconfig is None:
            raise ValueError("qat.Linear requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary linear layer
        """Fake-quantize the weight, run linear, fake-quantize the output."""
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.linear(x, w_q, self.bias)
        return cast("Tensor", self.activation_post_process(y))

    @classmethod
    def from_float(cls, mod: nn.Module) -> Linear:
        """Build a QAT ``Linear`` from a float one (shares the trained weight)."""
        lin = cast("nn.Linear", mod)
        qat = cls(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            qconfig=cast("QConfig", mod.qconfig),  # set by prepare_qat
        )
        # Adopt the trained float weight/bias directly — the prepare_qat
        # deep-copy already gave this module tree independent Parameters.
        qat.weight = lin.weight
        if lin.bias is not None:
            qat.bias = lin.bias
        return qat


class LinearReLU(Linear):
    """Quantization-aware fused ``Linear`` + ``ReLU`` (trainable, fake-quant).

    Behaves like :class:`Linear`, but the activation fake-quant observes the range
    *after* the fused ReLU, so the calibrated output grid reflects the true
    (non-negative) inference range.  Built from a fused float
    :class:`~lucid.nn.intrinsic.LinearReLU` by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` folds it
    into a single quantized :class:`~lucid.nn.quantized.LinearReLU`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default=True
        Whether to add a learnable (float) bias term.
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.

    Notes
    -----
    The weight and the post-ReLU output are both fake-quantized every forward via a
    straight-through estimator, keeping the float weight fully trainable.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary linear layer
        """Fake-quantize the weight, run linear, ReLU, fake-quantize the output."""
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.relu(F.linear(x, w_q, self.bias))
        return cast("Tensor", self.activation_post_process(y))

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> "LinearReLU":
        """Build from a fused float ``nni.LinearReLU`` (its inner linear)."""
        inner = cast("nn.Sequential", mod)[0]
        inner.qconfig = mod.qconfig
        return cast("LinearReLU", super().from_float(inner))
