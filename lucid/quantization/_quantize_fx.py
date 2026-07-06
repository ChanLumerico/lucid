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
    r"""Graph-mode prepare — auto-wrap the quant boundaries and insert observers.

    ``prepare_fx`` is the graph-mode entry point of static PTQ, the counterpart
    to eager :func:`~lucid.quantization.prepare`. It removes the one piece of
    manual bookkeeping the eager flow demands: instead of hand-placing a
    ``QuantStub`` at the model input and a ``DeQuantStub`` at the output, it wraps
    the model in a :class:`~lucid.nn.quantized.QuantWrapper` so the input is
    quantized and the output dequantized automatically, then runs the ordinary
    eager :func:`~lucid.quantization.prepare` underneath to attach activation
    observers. From there the workflow is identical — run calibration data
    through the returned model, then :func:`convert_fx`.

    The graph-mode payoff is downstream: because every quantized replacement is
    an ordinary composite, the converted model traces cleanly through
    ``lucid.compile``, letting the dispatch-bound quantized graph fuse into a
    single executable — the regime where quantization's inference win is
    structurally permanent. (True FX-style observer insertion into a *traced*
    graph would need C++ TraceGraph mutation; quantizing at the module level and
    letting ``compile`` capture it reaches the same result.)

    Parameters
    ----------
    model : nn.Module
        The float model — **no** manual ``QuantStub`` / ``DeQuantStub`` needed;
        the wrapper adds the boundaries.
    qconfig_mapping : QConfig or QConfigMapping, optional
        The quantization recipe, forwarded to the underlying
        :func:`~lucid.quantization.prepare`; ``None`` (default) uses the
        static-PTQ default mapping.
    example_inputs : Tensor, optional
        Reserved for a future trace-time insertion path; ignored on the current
        composite path (accepted for API compatibility).
    inplace : bool, default False
        If ``True`` mutate and return ``model`` itself; if ``False`` (default)
        prepare a deep copy.

    Returns
    -------
    nn.Module
        The prepared, ``eval``-mode model (input/output boundaries wrapped,
        observers attached). Calibrate it, then call :func:`convert_fx`.

    Notes
    -----
    - The only behavioral difference from eager :func:`~lucid.quantization.prepare`
      is the automatic ``QuantWrapper`` — everything else (observer types,
      calibration semantics, deep-copy default) is shared.
    - ``example_inputs`` is a placeholder today; passing it changes nothing.
    - Nothing is quantized yet — calibrate before :func:`convert_fx`, exactly as
      in the eager flow.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare_fx(model)          # no QuantStub placed by hand
    >>> for _ in range(8):                       # calibrate
    ...     _ = prepared(lucid.randn(32, 128))
    >>> qmodel = Q.convert_fx(prepared)
    >>> compiled = lucid.compile(qmodel)         # quantized graph fuses cleanly

    See Also
    --------
    lucid.quantization.convert_fx : The graph-mode bake step.
    lucid.quantization.prepare : The eager-mode counterpart (manual boundaries).
    lucid.nn.quantized.QuantWrapper : The input/output boundary wrapper used here.
    """
    del example_inputs  # reserved (composite path needs no trace here)
    import lucid.nn.quantized as nnq

    wrapped = nnq.QuantWrapper(model)
    return prepare(wrapped, qconfig_mapping, inplace=inplace)


def convert_fx(model: nn.Module, inplace: bool = False) -> nn.Module:
    r"""Graph-mode convert — bake a calibrated model into quantized modules.

    ``convert_fx`` is the graph-mode counterpart of
    :func:`~lucid.quantization.convert` and the final step of the graph-mode
    static-PTQ pipeline (``prepare_fx`` → *calibrate* → ``convert_fx``). It swaps
    each observed float module for its quantized inference equivalent, quantizing
    weights to int8 codes and folding the calibrated activation qparams into each
    layer. Because :func:`prepare_fx` already wrapped the quant boundaries, the
    result is a plain, self-contained quantized model that both runs directly and
    traces cleanly through ``lucid.compile`` — where the quantized graph fuses
    into a single executable.

    This is a thin pass-through over :func:`~lucid.quantization.convert`, so it
    shares all of its behavior: MLX-backend routing of the Linear family to the
    real weight-only ``quantized_matmul`` GEMM (~``3.15x`` decode speed-up at
    ``M = 1``, ~``3.55x`` smaller weights) when that backend is active, the
    reference dequant path otherwise, and the same deep-copy-by-default contract.

    Parameters
    ----------
    model : nn.Module
        A model returned by :func:`prepare_fx` and then run through calibration,
        so its observed submodules carry frozen activation qparams.
    inplace : bool, default False
        If ``True`` mutate and return ``model`` itself; if ``False`` (default)
        convert a deep copy and leave the calibrated model intact.

    Returns
    -------
    nn.Module
        The converted, ``eval``-mode quantized model — ready to run or hand to
        ``lucid.compile``.

    Notes
    -----
    - Delegates to :func:`~lucid.quantization.convert`; the only reason to call
      this variant is symmetry with :func:`prepare_fx` (the wrapping already
      happened at prepare time, so nothing extra is needed here).
    - Convert only *after* calibration — an uncalibrated model bakes in
      meaningless activation qparams.
    - The MLX compute win lands in the inference / generation regime and fades
      toward parity in compute-bound training GEMMs; force exact reference
      numerics with ``backends.quantized.engine = "reference"``.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare_fx(model)
    >>> for _ in range(8):
    ...     _ = prepared(lucid.randn(32, 128))
    >>> qmodel = Q.convert_fx(prepared)
    >>> compiled = lucid.compile(qmodel)         # single fused quantized graph

    See Also
    --------
    lucid.quantization.prepare_fx : The graph-mode observer-insertion step.
    lucid.quantization.convert : The eager-mode bake step this delegates to.
    lucid.compile : Fuses the converted quantized graph into one executable.
    """
    return convert(model, inplace=inplace)
