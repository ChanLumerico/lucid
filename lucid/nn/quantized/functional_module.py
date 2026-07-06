"""Quantizable functional merges — ``add`` / ``mul`` / ``cat`` / ``add_relu``.

Element-wise merges (residual skip-adds, concatenations) carry no weight, so
the module-swap machinery never sees them — yet a residual network cannot be
quantized end-to-end unless the merge output is observed and requantized to a
consistent grid.  A model uses :class:`FloatFunctional` in place of a bare
``x + y`` / :func:`lucid.cat` so ``prepare`` can attach an observer to the
result; ``convert`` then swaps it for :class:`QFunctional`, which fake-quantizes
each merge output to the calibrated activation grid (design B — float-carried).
"""

from typing import TYPE_CHECKING, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import activation_qparams
from lucid.quantization._functional import fake_quantize
from lucid.quantization._qscheme import QDtype, quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class FloatFunctional(nn.Module):
    """Float-domain quantizable merge ops; ``prepare`` observes their output.

    Drop into a model in place of raw ``x + y`` / :func:`lucid.cat` so the merge
    result is calibrated.  Until ``prepare`` attaches an ``activation_post_process``
    observer these stay plain float ops (the ``_obs`` passthrough).
    """

    def _obs(self, x: Tensor) -> Tensor:
        """Feed the result to the attached observer (identity before ``prepare``)."""
        app = getattr(self, "activation_post_process", None)
        return cast("Tensor", app(x)) if app is not None else x

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        """Observed ``x + y`` (a residual skip-add)."""
        return self._obs(x + y)

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        """Observed ``x * y``."""
        return self._obs(x * y)

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        """Observed ``relu(x + y)`` — the fused residual-add of a ResNet block."""
        return self._obs(F.relu(x + y))

    def cat(self, tensors: list[Tensor], dim: int = 0) -> Tensor:
        """Observed concatenation along ``dim``."""
        return self._obs(lucid.cat(tensors, dim=dim))

    def add_scalar(self, x: Tensor, scalar: float) -> Tensor:
        """Observed ``x + scalar``."""
        return self._obs(x + scalar)

    def mul_scalar(self, x: Tensor, scalar: float) -> Tensor:
        """Observed ``x * scalar``."""
        return self._obs(x * scalar)


class QFunctional(nn.Module):
    """Quantized merge ops — fake-quantizes each result to the calibrated grid."""

    scale: Tensor
    zero_point: Tensor

    def __init__(
        self, scale: Tensor, zero_point: Tensor, qdtype: QDtype = quint8
    ) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.qdtype = qdtype

    def _q(self, x: Tensor) -> Tensor:
        """Fake-quantize ``x`` to the calibrated output grid."""
        return fake_quantize(
            x, self.scale, self.zero_point, self.qdtype.quant_min, self.qdtype.quant_max
        )

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        """Quantized ``x + y``."""
        return self._q(x + y)

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        """Quantized ``x * y``."""
        return self._q(x * y)

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        """Quantized ``relu(x + y)``."""
        return self._q(F.relu(x + y))

    def cat(self, tensors: list[Tensor], dim: int = 0) -> Tensor:
        """Quantized concatenation along ``dim``."""
        return self._q(lucid.cat(tensors, dim=dim))

    def add_scalar(self, x: Tensor, scalar: float) -> Tensor:
        """Quantized ``x + scalar``."""
        return self._q(x + scalar)

    def mul_scalar(self, x: Tensor, scalar: float) -> Tensor:
        """Quantized ``x * scalar``."""
        return self._q(x * scalar)

    @classmethod
    def from_float(cls, mod: nn.Module) -> QFunctional:
        """Build from a calibrated :class:`FloatFunctional` (reads its observer)."""
        scale, zero_point, qdtype = activation_qparams(mod)
        return cls(scale, zero_point, qdtype)
