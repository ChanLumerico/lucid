"""Fused (*intrinsic*) float modules produced by ``fuse_modules``.

Fusing a ``Conv``/``Linear`` with the activation (and, for conv, the
BatchNorm folded into the weight) that follows it does two things for
quantization: it lets a **single** activation observer see the *post*-ReLU
output — so the quantized layer's output grid is chosen for the actual
(non-negative) inference range — and it removes the intermediate float
tensors.  Each intrinsic module is a thin :class:`~lucid.nn.Sequential`
subclass; ``convert`` maps it to the matching ``nn.quantized`` fused layer.
"""

import lucid.nn as nn


class _FusedModule(nn.Sequential):
    """Base tag for fused float modules (a :class:`~lucid.nn.Sequential`)."""


class ConvReLU1d(_FusedModule):
    """``Conv1d`` followed by ``ReLU``."""

    def __init__(self, conv: nn.Conv1d, relu: nn.ReLU) -> None:
        super().__init__(conv, relu)


class ConvReLU2d(_FusedModule):
    """``Conv2d`` followed by ``ReLU`` (BatchNorm already folded in)."""

    def __init__(self, conv: nn.Conv2d, relu: nn.ReLU) -> None:
        super().__init__(conv, relu)


class ConvReLU3d(_FusedModule):
    """``Conv3d`` followed by ``ReLU``."""

    def __init__(self, conv: nn.Conv3d, relu: nn.ReLU) -> None:
        super().__init__(conv, relu)


class LinearReLU(_FusedModule):
    """``Linear`` followed by ``ReLU``."""

    def __init__(self, linear: nn.Linear, relu: nn.ReLU) -> None:
        super().__init__(linear, relu)
