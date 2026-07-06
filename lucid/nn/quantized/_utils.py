"""Shared ``from_float`` helpers for quantized modules.

Quantizing a calibrated (PTQ) or trained (QAT) float module is the same
two-step recipe for every weighted layer: derive the **weight** qparams and
read the **activation** qparams, then bake them in.  These helpers centralise
that so ``Linear`` / the ``Conv`` family stay tiny and cover both the PTQ and
QAT source modules.
"""

import warnings
from typing import TYPE_CHECKING, cast

import lucid
from lucid.quantization._functional import quantize

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    import lucid.nn as nn
    from lucid.nn.parameter import Parameter
    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization._qscheme import QDtype
    from lucid.quantization.observer import ObserverBase
    from lucid.quantization.qconfig import QConfig


def quantize_weight(mod: nn.Module) -> tuple[Tensor, Tensor, Tensor, int]:
    """Quantize a module's weight, returning ``(codes, scale, zero_point, ch_axis)``.

    Works for both sources: a **QAT** module carries a ``weight_fake_quant``
    whose observer already saw the (trained) weight — reuse its qparams; a
    **PTQ** float module instead runs its ``qconfig``'s weight observer over
    the static weight now.
    """
    weight = cast("Tensor | Parameter", mod.weight)
    wfq = getattr(mod, "weight_fake_quant", None)
    if wfq is not None:
        wfq_m = cast("FakeQuantize", wfq)
        wfq_m(weight)  # refresh the observer on the current weight
        scale, zero_point = wfq_m.calculate_qparams()
        qdtype, ch_axis = wfq_m.qdtype, wfq_m.ch_axis
    else:
        qconfig = cast("QConfig", mod.qconfig)
        wobs = cast("ObserverBase", qconfig.weight())
        wobs(weight)
        scale, zero_point = wobs.calculate_qparams()
        qdtype, ch_axis = wobs.qdtype, wobs.ch_axis
    axis = ch_axis if ch_axis is not None else 0
    codes = quantize(weight, scale, zero_point, qdtype, ch_axis=ch_axis)
    return codes, scale, zero_point, axis


def activation_qparams(mod: nn.Module) -> tuple[Tensor, Tensor, QDtype]:
    """Read the output ``(scale, zero_point, qdtype)`` from a module's observer.

    ``activation_post_process`` is a bare observer for PTQ and a
    :class:`FakeQuantize` for QAT — both expose ``calculate_qparams`` / ``qdtype``.
    """
    obs = cast("ObserverBase", mod.activation_post_process)
    # A PTQ observer that never saw calibration data keeps its ±inf seed, which
    # collapses ``scale`` to ``eps`` and makes the quantized output ~0.  Warn
    # loudly rather than fail silently.  (QAT's FakeQuantize has no ``min_val``,
    # so this only fires on the uncalibrated static path.)
    min_val = getattr(obs, "min_val", None)
    if min_val is not None and bool(lucid.isinf(cast("Tensor", min_val)).any().item()):
        warnings.warn(
            f"Quantizing {type(mod).__name__} whose activation observer never saw "
            "data — run calibration through the prepared model before convert(), "
            "or the quantized output collapses to a near-zero grid.",
            stacklevel=3,
        )
    scale, zero_point = obs.calculate_qparams()
    return scale, zero_point, obs.qdtype
