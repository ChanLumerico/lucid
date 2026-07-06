"""``lucid.quantization`` — post-training quantization, QAT, and quantized inference.

This sub-package brings weights and activations from ``float32`` down to
low-precision integer grids (int8 / int4), the top capability lever for
inference and serving on Apple Silicon.  It mirrors the reference
framework's quantization surface while keeping a single canonical path
per name (H8): the *workflow* API lives here (observers, ``QConfig``,
``prepare`` / ``convert`` / ``quantize_dynamic`` / ``prepare_qat``), and
the quantized *module* layer lives under ``lucid.nn.quantized`` /
``lucid.nn.qat`` / ``lucid.nn.intrinsic``.

Representation follows the *sidecar* design (B): a quantized weight is an
ordinary integer :class:`~lucid.Tensor` plus float ``scale`` /
``zero_point`` buffers held by the owning module — no dedicated quantized
tensor subtype and no new engine dtype.

**Phase 0 surface (this commit):** the schemes / dtypes / parameter
container and the three arithmetic primitives (:func:`quantize`,
:func:`dequantize`, :func:`fake_quantize`).  Observers, ``QConfig``, the
eager / dynamic / graph workflows, and the real low-precision GEMM land
in later phases.
"""

from lucid.quantization._functional import dequantize, fake_quantize, quantize
from lucid.quantization._qparams import QParams, calculate_qparams
from lucid.quantization._qscheme import (
    QDtype,
    QScheme,
    per_channel_affine,
    per_channel_symmetric,
    per_tensor_affine,
    per_tensor_symmetric,
    qint4,
    qint8,
    qint32,
    quint8,
)
from lucid.quantization._fake_quantize import FakeQuantize
from lucid.quantization.observer import (
    FixedQParamsObserver,
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    NoopObserver,
    ObserverBase,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from lucid.quantization.qconfig import (
    QConfig,
    QConfigMapping,
    get_default_qat_qconfig,
    get_default_qat_qconfig_mapping,
    get_default_qconfig,
    get_default_qconfig_mapping,
)
from lucid.quantization._quantize import (
    convert,
    prepare,
    prepare_qat,
    quantize_dynamic,
)
from lucid.quantization._quantize_fx import convert_fx, prepare_fx
from lucid.quantization.fuse_modules import fuse_modules

__all__ = [
    # schemes
    "QScheme",
    "per_tensor_affine",
    "per_tensor_symmetric",
    "per_channel_affine",
    "per_channel_symmetric",
    # quantized dtypes
    "QDtype",
    "qint8",
    "quint8",
    "qint32",
    "qint4",
    # parameters
    "QParams",
    "calculate_qparams",
    # primitives
    "quantize",
    "dequantize",
    "fake_quantize",
    # observers
    "ObserverBase",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "HistogramObserver",
    "FixedQParamsObserver",
    "PlaceholderObserver",
    "NoopObserver",
    # QAT fake-quant
    "FakeQuantize",
    # config
    "QConfig",
    "QConfigMapping",
    "get_default_qconfig",
    "get_default_qat_qconfig",
    "get_default_qconfig_mapping",
    "get_default_qat_qconfig_mapping",
    # eager static PTQ workflow
    "prepare",
    "convert",
    "fuse_modules",
    # dynamic PTQ
    "quantize_dynamic",
    # QAT
    "prepare_qat",
    # graph-mode (compile-oriented)
    "prepare_fx",
    "convert_fx",
]


def __dir__() -> list[str]:
    """Restrict introspection to the public API — private ``_*`` implementation
    modules (``_functional`` / ``_qscheme`` / …) stay out of autocomplete."""
    return list(__all__)
