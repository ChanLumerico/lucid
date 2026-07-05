"""``FakeQuantize`` — an observer paired with a straight-through fake-quant.

During quantization-aware training a :class:`FakeQuantize` module sits on
a weight or activation: on every forward it (optionally) lets its
observer refresh ``(scale, zero_point)`` from the live statistics, then
(optionally) applies :func:`~lucid.quantization.fake_quantize` so the
downstream graph sees quantization rounding while gradients still flow
via the straight-through estimator.

The two toggles — ``enable_observer`` / ``enable_fake_quant`` — mirror the
reference framework and drive the standard QAT schedule: observe+fake-quant
early, then freeze the observer (and later the BN stats) as training
converges.
"""

import functools
from typing import TYPE_CHECKING, override

import lucid
import lucid.nn as nn
from lucid.quantization._functional import fake_quantize
from lucid.quantization._qscheme import QDtype, QScheme
from lucid.quantization.observer import MovingAverageMinMaxObserver, ObserverBase

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class FakeQuantize(nn.Module):
    """Observer + straight-through fake-quantization, toggleable for QAT.

    Parameters
    ----------
    observer : type[ObserverBase], default MovingAverageMinMaxObserver
        Observer class instantiated to track statistics.
    **observer_kwargs : object
        Forwarded to the observer constructor (``qscheme`` / ``qdtype`` /
        ``ch_axis`` / …).
    """

    scale: Tensor
    zero_point: Tensor

    def __init__(
        self,
        observer: type[ObserverBase] = MovingAverageMinMaxObserver,
        **observer_kwargs: object,
    ) -> None:
        super().__init__()
        self.activation_post_process: ObserverBase = observer(**observer_kwargs)  # type: ignore[arg-type]
        self.qdtype: QDtype = self.activation_post_process.qdtype
        self.qscheme: QScheme = self.activation_post_process.qscheme
        self.ch_axis: int | None = self.activation_post_process.ch_axis
        self.register_buffer("scale", lucid.tensor(1.0))
        self.register_buffer("zero_point", lucid.tensor(0.0))
        self._observer_enabled = True
        self._fake_quant_enabled = True

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # fake-quant forward is unary
        """Refresh qparams (if observing) then fake-quantize (if enabled)."""
        if self._observer_enabled:
            self.activation_post_process(x)
            scale, zero_point = self.activation_post_process.calculate_qparams()
            self.register_buffer("scale", scale)
            self.register_buffer("zero_point", zero_point)
        if self._fake_quant_enabled:
            return fake_quantize(
                x,
                self.scale,
                self.zero_point,
                self.qdtype.quant_min,
                self.qdtype.quant_max,
                self.ch_axis,
            )
        return x

    def enable_observer(self, enabled: bool = True) -> FakeQuantize:
        """Enable/disable statistic collection; returns ``self`` for chaining."""
        self._observer_enabled = enabled
        return self

    def disable_observer(self) -> FakeQuantize:
        """Freeze the observer (stop refreshing ``scale`` / ``zero_point``)."""
        return self.enable_observer(False)

    def enable_fake_quant(self, enabled: bool = True) -> FakeQuantize:
        """Enable/disable the fake-quant transform; returns ``self``."""
        self._fake_quant_enabled = enabled
        return self

    def disable_fake_quant(self) -> FakeQuantize:
        """Pass activations through unchanged (observer may still run)."""
        return self.enable_fake_quant(False)

    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        """Delegate to the wrapped observer."""
        return self.activation_post_process.calculate_qparams()

    @classmethod
    def with_args(cls, **kwargs: object) -> functools.partial[FakeQuantize]:
        """Return a zero-arg factory building this module with ``kwargs``."""
        return functools.partial(cls, **kwargs)  # type: ignore[arg-type]
