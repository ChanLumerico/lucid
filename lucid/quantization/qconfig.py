"""``QConfig`` — the (activation, weight) observer/fake-quant recipe.

A :class:`QConfig` pairs the observer recipe for **activations** with the
one for **weights**; a :class:`QConfigMapping` then says which QConfig
applies to which part of a model (globally, by module type, or by
qualified name).  This is the reference framework's structure, and it is
the analogue of the AMP autocast policy: a small declarative object that
``prepare`` / ``prepare_qat`` consult when inserting observers.

The ``activation`` / ``weight`` fields hold **factories** (zero-arg
callables, typically produced by ``Observer.with_args(...)`` /
``FakeQuantize.with_args(...)``) so one QConfig can be instantiated fresh
at every insertion site.
"""

from typing import Callable, NamedTuple

import lucid.nn as nn
from lucid.quantization._qscheme import (
    per_channel_symmetric,
    per_tensor_affine,
    qint8,
    quint8,
)
from lucid.quantization._fake_quantize import FakeQuantize
from lucid.quantization.observer import (
    HistogramObserver,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
)

_Factory = Callable[..., nn.Module]


class QConfig(NamedTuple):
    """A pair of observer / fake-quant factories for activations and weights.

    The atomic quantization recipe: ``prepare`` / ``prepare_qat`` call the two
    factories to build a fresh observer (PTQ) or :class:`FakeQuantize` (QAT) at
    every insertion site.  A :class:`QConfigMapping` then decides *which*
    QConfig applies to *which* module.  Both fields are zero-arg callables —
    typically produced by ``Observer.with_args(...)`` /
    ``FakeQuantize.with_args(...)`` — so one recipe instantiates independent
    modules across the whole network.

    Attributes
    ----------
    activation : Callable[..., nn.Module]
        Factory for the **activation** observer / fake-quant.
    weight : Callable[..., nn.Module]
        Factory for the **weight** observer / fake-quant.
    """

    activation: _Factory
    weight: _Factory


def get_default_qconfig() -> QConfig:
    """Default **static PTQ** config.

    Activations use a per-tensor-affine ``quint8`` histogram observer
    (high accuracy on heavy-tailed activations); weights use a
    per-channel-symmetric ``qint8`` min/max observer — the standard,
    accuracy-preserving combination for convolution / linear layers.
    """
    return QConfig(
        activation=HistogramObserver.with_args(
            qscheme=per_tensor_affine, qdtype=quint8
        ),
        weight=PerChannelMinMaxObserver.with_args(
            ch_axis=0, qscheme=per_channel_symmetric, qdtype=qint8
        ),
    )


def get_default_qat_qconfig() -> QConfig:
    """Default **QAT** config — the same schemes wrapped in :class:`FakeQuantize`.

    Activations use a moving-average per-tensor-affine ``quint8`` fake-quant;
    weights use a per-channel-symmetric ``qint8`` fake-quant so the training
    graph simulates the eventual quantized inference numerics.
    """
    return QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            qscheme=per_tensor_affine,
            qdtype=quint8,
        ),
        weight=FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            ch_axis=0,
            qscheme=per_channel_symmetric,
            qdtype=qint8,
        ),
    )


class QConfigMapping:
    """Resolves which :class:`QConfig` applies to each module.

    Resolution order (most specific wins): qualified **module name** →
    **module type** → **global**.  Setters return ``self`` for fluent
    chaining, mirroring the reference framework.
    """

    def __init__(self) -> None:
        self._global: QConfig | None = None
        self._object_type: dict[type, QConfig | None] = {}
        self._module_name: dict[str, QConfig | None] = {}

    def set_global(self, qconfig: QConfig | None) -> QConfigMapping:
        """Set the fallback QConfig applied when nothing more specific matches."""
        self._global = qconfig
        return self

    def set_object_type(
        self, module_type: type, qconfig: QConfig | None
    ) -> QConfigMapping:
        """Bind a QConfig to every module of ``module_type``."""
        self._object_type[module_type] = qconfig
        return self

    def set_module_name(self, name: str, qconfig: QConfig | None) -> QConfigMapping:
        """Bind a QConfig to the module at the qualified ``name``."""
        self._module_name[name] = qconfig
        return self

    def get_qconfig(self, module_type: type, module_name: str) -> QConfig | None:
        """Return the QConfig for a module, most-specific match first."""
        if module_name in self._module_name:
            return self._module_name[module_name]
        if module_type in self._object_type:
            return self._object_type[module_type]
        return self._global


def get_default_qconfig_mapping() -> QConfigMapping:
    """Build the default **static-PTQ** :class:`QConfigMapping`.

    Sets :func:`get_default_qconfig` as the global fallback, so every eligible
    module is quantized with the standard histogram-activation /
    per-channel-weight recipe unless overridden by type or name.

    Returns
    -------
    QConfigMapping
        A mapping whose global QConfig is the default PTQ config.
    """
    return QConfigMapping().set_global(get_default_qconfig())


def get_default_qat_qconfig_mapping() -> QConfigMapping:
    """Build the default **QAT** :class:`QConfigMapping`.

    Sets :func:`get_default_qat_qconfig` as the global fallback, so every
    eligible module trains with :class:`FakeQuantize` on its activations and
    weights unless overridden by type or name.

    Returns
    -------
    QConfigMapping
        A mapping whose global QConfig is the default QAT config.
    """
    return QConfigMapping().set_global(get_default_qat_qconfig())
