"""Graph-mode quantization: ``prepare_fx`` / ``convert_fx``.

The graph-oriented counterpart to eager ``prepare`` / ``convert``.  Two
differences from the eager flow:

1. **No manual boundaries** — the model is auto-wrapped so its input is
   quantized and its output dequantized; you don't hand-place ``QuantStub`` /
   ``DeQuantStub``.
2. **Compile-ready** — the converted model traces cleanly through
   ``lucid.compile`` (the quantized layers are ordinary composites), so the
   dispatch-bound quantized graph fuses into a single executable — the regime
   where quantization's inference win is structurally permanent.

Automatic observer insertion into the *traced* graph (true FX) would need
C++ TraceGraph mutation; the composite path here reaches the same result by
quantizing at the module level and letting ``compile`` capture it.
"""

from typing import TYPE_CHECKING

from lucid.quantization._quantize import convert, prepare

if TYPE_CHECKING:
    import lucid.nn as nn

    from lucid._tensor.tensor import Tensor
    from lucid.quantization.qconfig import QConfig, QConfigMapping


def prepare_fx(
    model: nn.Module,
    qconfig_mapping: QConfig | QConfigMapping | None = None,
    example_inputs: Tensor | None = None,
    inplace: bool = False,
) -> nn.Module:
    """Graph-mode prepare — auto-wrap quant boundaries and insert observers.

    Parameters
    ----------
    model : nn.Module
        Float model (no manual ``QuantStub`` needed).
    qconfig_mapping : QConfig or QConfigMapping, optional
        Quantization recipe; defaults to the static-PTQ default.
    example_inputs : Tensor, optional
        Reserved for future trace-time insertion; unused on the composite path.
    inplace : bool, default False
        Mutate in place instead of deep-copying.

    Returns
    -------
    nn.Module
        Prepared model — calibrate, then :func:`convert_fx`.
    """
    del example_inputs  # reserved (composite path needs no trace here)
    import lucid.nn.quantized as nnq

    wrapped = nnq.QuantWrapper(model)
    return prepare(wrapped, qconfig_mapping, inplace=inplace)


def convert_fx(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Graph-mode convert — bake a calibrated model into quantized modules.

    The graph-mode counterpart of :func:`~lucid.quantization.convert`: it swaps
    each observed float module for its quantized inference equivalent, folding in
    the calibrated qparams.  The result is a plain quantized model, ready to run
    or hand to ``lucid.compile``.

    Parameters
    ----------
    model : nn.Module
        A model returned by :func:`prepare_fx` and then calibrated.
    inplace : bool, default False
        Mutate ``model`` in place instead of deep-copying it first.

    Returns
    -------
    nn.Module
        The quantized model.
    """
    return convert(model, inplace=inplace)
