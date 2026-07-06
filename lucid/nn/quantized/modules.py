"""Quantization boundary markers and their converted forms.

``QuantStub`` / ``DeQuantStub`` are placed in a **float** model to mark
where the quantized region begins and ends.  ``prepare`` attaches an
activation observer to the ``QuantStub``; ``convert`` then swaps the stubs
for their runtime forms:

* :class:`Quantize` — fake-quantizes its input to the calibrated
  activation ``(scale, zero_point)``, i.e. the entry into the quantized
  region.
* :class:`DeQuantize` — the exit.  Under Lucid's sidecar representation
  (design B) activations are carried as (fake-quantized) ``float32``
  throughout, so dequantization is the identity.
"""

from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
from lucid.quantization._fake_quantize import FakeQuantize
from lucid.quantization._functional import fake_quantize
from lucid.quantization._qscheme import QDtype, quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.quantization.observer import ObserverBase


class QuantStub(nn.Module):
    """Marks the float→quantized boundary in a to-be-quantized model.

    Identity at float / calibration time; ``prepare`` attaches an activation
    observer and ``convert`` replaces it with :class:`Quantize`.
    """

    def __init__(self, qconfig: object = None) -> None:
        super().__init__()
        if qconfig is not None:
            self.qconfig = qconfig

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary boundary marker
        """Identity for PTQ (a hook observes); fake-quant for QAT.

        In QAT ``prepare_qat`` attaches a :class:`FakeQuantize` as
        ``activation_post_process`` — apply it so the input is fake-quantized
        during training.  In PTQ it's a bare observer fed by a forward hook,
        so the stub stays an identity.
        """
        app = getattr(self, "activation_post_process", None)
        if isinstance(app, FakeQuantize) and lucid.is_floating_point(x):
            return cast("Tensor", app(x))
        return x


class DeQuantStub(nn.Module):
    """Marks the quantized→float boundary in a to-be-quantized model.

    The exit-side counterpart of :class:`QuantStub`: place it where the
    quantized region ends and float compute resumes. It is an identity at
    float / calibration time; :func:`lucid.quantization.convert` replaces it
    with the runtime :class:`DeQuantize` marker. Under Lucid's sidecar
    representation (design B) activations are carried as (fake-quantized) float
    throughout, so this marker never changes tensor values.

    Parameters
    ----------
    qconfig : object, optional
        Quantization config attached to this boundary. When ``None`` (the
        default) no ``qconfig`` attribute is set and the module inherits the
        surrounding model's config during ``prepare`` / ``convert``.
    """

    def __init__(self, qconfig: object = None) -> None:
        super().__init__()
        if qconfig is not None:
            self.qconfig = qconfig

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary boundary marker
        """Identity."""
        return x


class Quantize(nn.Module):
    """Runtime entry into the quantized region — fake-quantizes the input.

    The converted form of a calibrated :class:`QuantStub`, produced by
    :func:`lucid.quantization.convert` / :meth:`from_float`. Each forward
    quantizes a real-valued activation to the calibrated ``(scale,
    zero_point)`` grid (integer inputs such as token indices are passed through
    unchanged). Under the sidecar representation (design B) the result is
    carried as fake-quantized float, so it has int8 numerics while staying a
    float tensor.

    Parameters
    ----------
    scale : Tensor
        Per-tensor activation scale from the calibrating observer.
    zero_point : Tensor
        Per-tensor activation zero-point from the calibrating observer.
    qdtype : QDtype, optional
        Quantized dtype whose ``quant_min`` / ``quant_max`` bound the grid.
        Defaults to :data:`~lucid.quantization.quint8`.
    """

    scale: Tensor
    zero_point: Tensor

    def __init__(
        self, scale: Tensor, zero_point: Tensor, qdtype: QDtype = quint8
    ) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.qdtype = qdtype

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary boundary marker
        """Quantize ``x`` to the calibrated activation grid (float-carried).

        Integer inputs (e.g. token indices) are passed through unchanged —
        only real-valued activations are quantized.
        """
        if not lucid.is_floating_point(x):
            return x
        return fake_quantize(
            x, self.scale, self.zero_point, self.qdtype.quant_min, self.qdtype.quant_max
        )

    @classmethod
    def from_float(cls, stub: nn.Module) -> Quantize:
        """Build from a calibrated :class:`QuantStub` (reads its observer)."""
        obs = cast("ObserverBase", stub.activation_post_process)
        scale, zero_point = obs.calculate_qparams()
        return cls(scale, zero_point, obs.qdtype)


class DeQuantize(nn.Module):
    """Runtime exit from the quantized region — identity under design B.

    The converted form of a :class:`DeQuantStub`, produced by
    :func:`lucid.quantization.convert` / :meth:`from_float`. It marks where the
    quantized region ends. Because Lucid's sidecar representation (design B)
    carries activations as (fake-quantized) float throughout, there is nothing
    to unpack: the forward is a plain identity and the module holds no state.

    Notes
    -----
    Takes no constructor arguments. It exists as an explicit boundary marker so
    the quantized graph is symmetric with :class:`Quantize` — useful for
    tooling that walks the module tree.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary boundary marker
        """Identity (activations are already float-carried)."""
        return x

    @classmethod
    def from_float(cls, stub: nn.Module) -> DeQuantize:
        """Build from a :class:`DeQuantStub` (no state to carry)."""
        return cls()


class QuantWrapper(nn.Module):
    """Wrap an arbitrary float model with ``QuantStub`` / ``DeQuantStub``.

    Lets a model that has no explicit quantization boundaries (e.g. a stock
    zoo model) be quantized: the input is quantized on entry and dequantized
    on exit, with the wrapped model in between.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.module = module
        self.dequant = DeQuantStub()

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary wrapper
        """Quantize the input, run the wrapped model, dequantize the output."""
        x = cast("Tensor", self.quant(x))
        x = cast("Tensor", self.module(x))
        return cast("Tensor", self.dequant(x))
