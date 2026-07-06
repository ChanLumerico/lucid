"""Eager-mode static post-training quantization: ``prepare`` → ``convert``.

The classic three-step flow:

1. :func:`prepare` — attach an activation observer (as a forward hook) to
   every quantizable module the :class:`QConfigMapping` selects.
2. *calibration* — the caller runs representative data through the prepared
   model so the observers collect activation ranges.
3. :func:`convert` — swap each calibrated float module for its quantized
   counterpart (int8 weights baked in, activation qparams frozen).

``nn.quantized`` is imported lazily inside the functions so this module can
live in ``lucid.quantization`` without an import cycle (``nn.quantized``
imports the quantization primitives back).
"""

import copy
import warnings
from typing import TYPE_CHECKING, Callable, cast

import lucid.nn as nn
from lucid.quantization._qscheme import QDtype, qint8
from lucid.quantization.qconfig import (
    QConfig,
    QConfigMapping,
    get_default_qat_qconfig_mapping,
    get_default_qconfig_mapping,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    _Hook = Callable[[object, tuple[Tensor, ...], Tensor | tuple[Tensor, ...]], None]
    _FromFloat = Callable[[nn.Module], nn.Module]
    _DynFromFloat = Callable[[nn.Module, QDtype], nn.Module]


def _observed_types() -> tuple[type, ...]:
    """Module types that receive an activation observer during ``prepare``."""
    import lucid.nn.intrinsic as nni
    import lucid.nn.quantized as nnq

    return (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nni.ConvReLU1d,
        nni.ConvReLU2d,
        nni.ConvReLU3d,
        nni.LinearReLU,
        nnq.QuantStub,
        nnq.FloatFunctional,
    )


def _quant_mapping() -> dict[type, _FromFloat]:
    """Map each float / QAT module type to its quantized ``from_float`` builder."""
    import lucid.nn.intrinsic as nni
    import lucid.nn.intrinsic.qat as nniqat
    import lucid.nn.qat as nnqat
    import lucid.nn.quantized as nnq
    from lucid.nn.intrinsic.qat.modules import convbnrelu2d_to_quantized

    return {
        nniqat.ConvBnReLU2d: convbnrelu2d_to_quantized,
        nn.Linear: nnq.Linear.from_float,
        nn.Conv1d: nnq.Conv1d.from_float,
        nn.Conv2d: nnq.Conv2d.from_float,
        nn.Conv3d: nnq.Conv3d.from_float,
        nn.Embedding: nnq.Embedding.from_float,
        nni.ConvReLU1d: nnq.ConvReLU1d.from_float,
        nni.ConvReLU2d: nnq.ConvReLU2d.from_float,
        nni.ConvReLU3d: nnq.ConvReLU3d.from_float,
        nni.LinearReLU: nnq.LinearReLU.from_float,
        # QAT modules → the same quantized inference layers.
        nnqat.Linear: nnq.Linear.from_float,
        nnqat.Conv1d: nnq.Conv1d.from_float,
        nnqat.Conv2d: nnq.Conv2d.from_float,
        nnqat.Conv3d: nnq.Conv3d.from_float,
        nnq.QuantStub: nnq.Quantize.from_float,
        nnq.DeQuantStub: nnq.DeQuantize.from_float,
        nnq.FloatFunctional: nnq.QFunctional.from_float,
    }


def _mlx_weight_bits(mod: nn.Module) -> int:
    """Weight bit-width (4/8) for the MLX fast path, read from the module's qconfig."""
    qcfg = getattr(mod, "qconfig", None)
    if qcfg is not None:
        bits = getattr(getattr(qcfg.weight(), "qdtype", None), "bits", None)
        if isinstance(bits, int) and bits in (4, 8):
            return bits
    return 8


def _mlx_group_size(in_features: int) -> int | None:
    """Largest valid MLX group size (32/64/128) dividing ``in_features``.

    ``mlx.core.quantize`` requires the quantized dimension to be a multiple of
    the group size, so a layer whose ``in_features`` divides none of the three
    supported sizes cannot use the group-wise kernel — return ``None`` and let
    the caller fall back to the dequantize path.
    """
    for gs in (64, 128, 32):
        if in_features % gs == 0:
            return gs
    return None


def _mlx_linear_builder(relu: bool) -> _FromFloat:
    """1-arg ``convert`` builder that routes a (fused) Linear to the real MLX GEMM.

    Weight-only group-wise quantization (``quantized_matmul``) — the actual
    speed + memory win, in exchange for W8A16 numerics (activations stay float).
    Selected only when ``backends.quantized.use_mlx()`` is true; layers whose
    ``in_features`` is not group-size-divisible fall back to the dequant path.
    """
    from lucid.nn.quantized import Linear as QLinear
    from lucid.nn.quantized import LinearReLU as QLinearReLU
    from lucid.nn.quantized import QuantizedLinearMLX

    dequant: _FromFloat = QLinearReLU.from_float if relu else QLinear.from_float

    def build(child: nn.Module) -> nn.Module:
        inner = cast("nn.Sequential", child)[0] if relu else child
        gs = _mlx_group_size(cast("nn.Linear", inner).in_features)
        if gs is None:
            return dequant(child)
        return QuantizedLinearMLX.from_float(
            inner, bits=_mlx_weight_bits(child), group_size=gs, relu=relu
        )

    return build


def _mlx_dynamic_builder() -> _DynFromFloat:
    """2-arg ``quantize_dynamic`` builder routing Linear to the real MLX GEMM."""
    import lucid.nn.quantized.dynamic as nnqd
    from lucid.nn.quantized import QuantizedLinearMLX

    def build(child: nn.Module, dtype: QDtype) -> nn.Module:
        gs = _mlx_group_size(cast("nn.Linear", child).in_features)
        if gs is None:
            return nnqd.Linear.from_float(child, dtype)
        return QuantizedLinearMLX.from_float(
            child, bits=dtype.bits, group_size=gs, relu=False
        )

    return build


def _make_observer_hook(obs: nn.Module) -> _Hook:
    """Forward hook that feeds a module's output to its activation observer."""

    def hook(
        _module: object,
        _args: tuple[Tensor, ...],
        output: Tensor | tuple[Tensor, ...],
    ) -> None:
        obs(cast("Tensor", output))

    return hook


def _as_mapping(qconfig: QConfig | QConfigMapping | None) -> QConfigMapping:
    """Coerce a bare QConfig (or ``None``) to a global :class:`QConfigMapping`."""
    if qconfig is None:
        return get_default_qconfig_mapping()
    if isinstance(qconfig, QConfig):
        return QConfigMapping().set_global(qconfig)
    return qconfig


def prepare(
    model: nn.Module,
    qconfig: QConfig | QConfigMapping | None = None,
    inplace: bool = False,
) -> nn.Module:
    """Insert activation observers ahead of calibration.

    Parameters
    ----------
    model : nn.Module
        Float model (with ``QuantStub`` / ``DeQuantStub`` boundaries).
    qconfig : QConfig or QConfigMapping, optional
        Quantization recipe; defaults to :func:`get_default_qconfig_mapping`.
    inplace : bool, default False
        Mutate ``model`` in place instead of deep-copying it.

    Returns
    -------
    nn.Module
        The prepared model — run calibration data through it, then
        :func:`convert`.
    """
    mapping = _as_mapping(qconfig)
    model = model if inplace else copy.deepcopy(model)
    model.eval()
    observed = _observed_types()
    n_matched = 0
    for name, mod in model.named_modules():
        qcfg = mapping.get_qconfig(type(mod), name)
        if qcfg is None or not isinstance(mod, observed):
            continue
        mod.qconfig = qcfg
        obs = qcfg.activation()
        mod.activation_post_process = obs
        mod.register_forward_hook(_make_observer_hook(obs))
        n_matched += 1
    if n_matched == 0:
        warnings.warn(
            "prepare(): the QConfigMapping matched no quantizable modules — "
            "convert() will leave this model unchanged.",
            stacklevel=2,
        )
    return model


def convert(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Swap calibrated float modules for their quantized replacements.

    When ``lucid.backends.quantized.use_mlx()`` is true (the MLX quantized
    engine is present), Linear-family layers are routed to the real weight-only
    ``quantized_matmul`` GEMM — the genuine speed + memory win — instead of the
    dequantize-to-float path.  Set ``backends.quantized.engine = "reference"``
    to force the exact W8A8 dequant numerics on every layer.
    """
    import lucid
    import lucid.nn.intrinsic as nni
    import lucid.nn.qat as nnqat
    import lucid.nn.quantized as nnq

    model = model if inplace else copy.deepcopy(model)
    mapping = _quant_mapping()
    if lucid.backends.quantized.use_mlx():
        mapping = dict(mapping)
        mapping[nn.Linear] = _mlx_linear_builder(relu=False)
        mapping[nni.LinearReLU] = _mlx_linear_builder(relu=True)
        mapping[nnqat.Linear] = _mlx_linear_builder(relu=False)
    # DeQuantStub + Embedding convert without calibration (no activation observer).
    _convert_recursive(model, mapping, always=(nnq.DeQuantStub, nn.Embedding))
    model.eval()
    return model


def _dynamic_mapping() -> dict[type, _DynFromFloat]:
    """Map each float module type to its dynamic-quant ``from_float`` builder."""
    import lucid.nn.quantized.dynamic as nnqd

    return {nn.Linear: nnqd.Linear.from_float, nn.LSTM: nnqd.LSTM.from_float}


def quantize_dynamic(
    model: nn.Module,
    qconfig_spec: set[type] | None = None,
    dtype: QDtype = qint8,
    inplace: bool = False,
) -> nn.Module:
    """Dynamically quantize a model — int8 weights, runtime-quantized activations.

    Unlike static PTQ this needs **no calibration**: every module whose type
    is in ``qconfig_spec`` (default ``{Linear, LSTM}``) is swapped for its
    dynamic counterpart, which measures the activation range per forward.

    Parameters
    ----------
    model : nn.Module
        Float model.
    qconfig_spec : set of module types, optional
        Which module types to quantize; defaults to ``{nn.Linear, nn.LSTM}``.
    dtype : QDtype, default ``qint8``
        Weight quantized dtype.
    inplace : bool, default False
        Mutate in place instead of deep-copying.
    """
    import lucid

    model = model if inplace else copy.deepcopy(model)
    model.eval()
    mapping = _dynamic_mapping()
    if lucid.backends.quantized.use_mlx():
        mapping = dict(mapping)
        mapping[nn.Linear] = _mlx_dynamic_builder()
    types = qconfig_spec if qconfig_spec is not None else set(mapping)
    if not any(type(m) in types for m in model.modules()):
        warnings.warn(
            "quantize_dynamic(): no module of a targeted type "
            f"({', '.join(sorted(t.__name__ for t in types))}) found — "
            "the model is returned unchanged.",
            stacklevel=2,
        )
    # Handle a bare top-level target (no parent to swap it in).
    build = mapping.get(type(model))
    if build is not None and type(model) in types:
        return build(model, dtype)
    _swap_dynamic(model, mapping, types, dtype)
    return model


def _swap_dynamic(
    module: nn.Module,
    mapping: dict[type, _DynFromFloat],
    types: set[type],
    dtype: QDtype,
) -> None:
    """Depth-first swap of each targeted module for its dynamic-quant form."""
    for name, child in list(module.named_children()):
        build = mapping.get(type(child))
        if build is not None and type(child) in types:
            module._modules[name] = build(child, dtype)
        else:
            _swap_dynamic(child, mapping, types, dtype)


def _qat_mapping() -> dict[type, _FromFloat]:
    """Map each float module type to its QAT (fake-quant, trainable) form."""
    import lucid.nn.qat as nnqat

    return {
        nn.Linear: nnqat.Linear.from_float,
        nn.Conv1d: nnqat.Conv1d.from_float,
        nn.Conv2d: nnqat.Conv2d.from_float,
        nn.Conv3d: nnqat.Conv3d.from_float,
    }


def prepare_qat(
    model: nn.Module,
    qconfig: QConfig | QConfigMapping | None = None,
    inplace: bool = False,
) -> nn.Module:
    """Insert fake-quant modules for quantization-aware training.

    Swaps weighted layers for their :mod:`lucid.nn.qat` counterparts (weight +
    output fake-quant) and attaches a fake-quant to every ``QuantStub``.  The
    returned model is left in ``train`` mode — fine-tune it, then
    :func:`convert` to the int8 inference model.

    Parameters
    ----------
    model : nn.Module
        Float model with ``QuantStub`` / ``DeQuantStub`` boundaries.
    qconfig : QConfig or QConfigMapping, optional
        QAT recipe; defaults to :func:`get_default_qat_qconfig_mapping`.
    inplace : bool, default False
        Mutate in place instead of deep-copying.
    """
    if qconfig is None:
        mapping = get_default_qat_qconfig_mapping()
    else:
        mapping = _as_mapping(qconfig)
    model = model if inplace else copy.deepcopy(model)
    model.train()
    _prepare_qat_recursive(model, mapping, _qat_mapping(), "")
    return model


def _prepare_qat_recursive(
    module: nn.Module,
    qconfig_mapping: QConfigMapping,
    qat_map: dict[type, _FromFloat],
    prefix: str,
) -> None:
    """Depth-first swap of weighted layers to QAT + fake-quant on ``QuantStub``."""
    import lucid.nn.quantized as nnq

    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        qcfg = qconfig_mapping.get_qconfig(type(child), full)
        build = qat_map.get(type(child))
        if qcfg is not None and build is not None:
            child.qconfig = qcfg
            module._modules[name] = build(child)
        elif qcfg is not None and isinstance(child, nnq.QuantStub):
            child.qconfig = qcfg
            child.activation_post_process = qcfg.activation()
        else:
            _prepare_qat_recursive(child, qconfig_mapping, qat_map, full)


def _convert_recursive(
    module: nn.Module, mapping: dict[type, _FromFloat], always: tuple[type, ...]
) -> None:
    """Depth-first swap of every convertible child in place."""
    for name, child in list(module.named_children()):
        build = mapping.get(type(child))
        convertible = build is not None and (
            isinstance(child, always) or hasattr(child, "activation_post_process")
        )
        if convertible and build is not None:
            # Assign through ``_modules`` (not ``setattr``) to preserve the
            # child's position: ``Module.__setattr__`` deletes then re-adds the
            # key, which would move it to the end and break ``Sequential`` order.
            module._modules[name] = build(child)
        else:
            _convert_recursive(child, mapping, always)
