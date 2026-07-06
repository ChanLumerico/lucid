"""Quantized activation modules — float activation + output fake-quant.

Under the sidecar design (B) activations are carried as (fake-quantized)
``float32``, so a *quantized* activation is simply the float activation followed
by a fake-quantize to the calibrated output grid.  ``convert`` swaps a calibrated
float ``nn.Sigmoid`` / ``nn.Hardswish`` / … for these so a standalone activation
inside a quantized region is requantized rather than left at full precision.
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import activation_qparams
from lucid.quantization._functional import fake_quantize
from lucid.quantization._qscheme import QDtype, quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class _QuantizedActivation(nn.Module):
    """Float activation then fake-quant to the calibrated ``(scale, zero_point)``."""

    scale: Tensor
    zero_point: Tensor

    def __init__(
        self, scale: Tensor, zero_point: Tensor, qdtype: QDtype = quint8
    ) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.qdtype = qdtype

    def _act(self, x: Tensor) -> Tensor:
        """Apply the float activation — overridden per subclass."""
        raise NotImplementedError

    def _configure(self, mod: nn.Module) -> None:
        """Copy activation-specific params (e.g. ``alpha``) from the float module."""

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary activation
        """Activate, then fake-quantize to the calibrated grid."""
        return fake_quantize(
            self._act(x),
            self.scale,
            self.zero_point,
            self.qdtype.quant_min,
            self.qdtype.quant_max,
        )

    @classmethod
    def from_float(cls, mod: nn.Module) -> "_QuantizedActivation":
        """Build from a calibrated float activation (reads its output observer)."""
        scale, zero_point, qdtype = activation_qparams(mod)
        obj = cls(scale, zero_point, qdtype)
        obj._configure(mod)
        return obj


class Sigmoid(_QuantizedActivation):
    """Quantized ``Sigmoid``."""

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Hardswish(_QuantizedActivation):
    """Quantized ``Hardswish``."""

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.hardswish(x)


class Hardsigmoid(_QuantizedActivation):
    """Quantized ``Hardsigmoid``."""

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.hardsigmoid(x)


class Tanh(_QuantizedActivation):
    """Quantized ``Tanh``."""

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class ELU(_QuantizedActivation):
    """Quantized ``ELU`` (carries the float module's ``alpha``)."""

    alpha: float = 1.0

    @override
    def _configure(self, mod: nn.Module) -> None:
        self.alpha = float(cast("float", getattr(mod, "alpha", 1.0)))

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.elu(x, self.alpha)


class LeakyReLU(_QuantizedActivation):
    """Quantized ``LeakyReLU`` (carries the float module's ``negative_slope``)."""

    negative_slope: float = 0.01

    @override
    def _configure(self, mod: nn.Module) -> None:
        self.negative_slope = float(cast("float", getattr(mod, "negative_slope", 0.01)))

    @override
    def _act(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)
