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
    """Quantization-aware ``Linear`` (weight + output fake-quant)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qconfig: QConfig | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        assert qconfig is not None, "qat.Linear requires a qconfig"
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
