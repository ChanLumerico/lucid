"""Quantized fused (intrinsic) modules — ``ConvReLU`` / ``LinearReLU``.

These are the quantized counterparts of the ``lucid.nn.intrinsic`` float
modules.  Each reuses the plain quantized layer and only overrides the
post-op activation hook to apply ReLU *before* the output is fake-quantized
— matching the calibration, where the fused module's activation observer
saw the post-ReLU range.  ``from_float`` delegates to the base builder via
``super()`` (so ``cls`` stays the fused subclass) after copying the fused
module's qconfig + observer onto its inner weighted child.
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized.conv import Conv1d, Conv2d, Conv3d
from lucid.nn.quantized.linear import Linear

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _wire_inner(mod: nn.Module) -> nn.Module:
    """Return the weighted child carrying the qparams.

    A **float** fused module is an ``nn.Sequential`` — copy its qconfig +
    observer onto ``seq[0]`` and return that.  A **QAT** fused module
    (``nn.qat.ConvReLU`` / ``LinearReLU``) is already a single weighted layer
    carrying ``weight_fake_quant`` + ``activation_post_process`` directly, so it
    is returned unchanged.
    """
    if not isinstance(mod, nn.Sequential):
        return mod
    inner = mod[0]
    inner.qconfig = mod.qconfig
    inner.activation_post_process = mod.activation_post_process
    return inner


class LinearReLU(Linear):
    """Quantized ``Linear`` + ``ReLU``."""

    @override
    def _activation(self, y: Tensor) -> Tensor:
        return F.relu(y)

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> LinearReLU:
        return cast("LinearReLU", super().from_float(_wire_inner(mod)))


class ConvReLU1d(Conv1d):
    """Quantized ``Conv1d`` + ``ReLU``."""

    @override
    def _activation(self, y: Tensor) -> Tensor:
        return F.relu(y)

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> ConvReLU1d:
        return cast("ConvReLU1d", super().from_float(_wire_inner(mod)))


class ConvReLU2d(Conv2d):
    """Quantized ``Conv2d`` + ``ReLU``."""

    @override
    def _activation(self, y: Tensor) -> Tensor:
        return F.relu(y)

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> ConvReLU2d:
        return cast("ConvReLU2d", super().from_float(_wire_inner(mod)))


class ConvReLU3d(Conv3d):
    """Quantized ``Conv3d`` + ``ReLU``."""

    @override
    def _activation(self, y: Tensor) -> Tensor:
        return F.relu(y)

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> ConvReLU3d:
        return cast("ConvReLU3d", super().from_float(_wire_inner(mod)))
