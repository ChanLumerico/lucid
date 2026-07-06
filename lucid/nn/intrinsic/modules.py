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
    """Fused float ``Conv1d`` + ``ReLU`` marker (a :class:`~lucid.nn.Sequential`).

    A thin :class:`~lucid.nn.Sequential` of ``[conv, relu]`` emitted by
    :func:`lucid.quantization.fuse_modules` so that a later activation observer sees the
    *post*-ReLU range, letting the eventual quantized layer pick an output grid for the
    true (non-negative) inference range.  It runs as plain float at this stage;
    :func:`lucid.quantization.convert` maps it to a fused quantized
    :class:`~lucid.nn.quantized.ConvReLU1d`.

    Parameters
    ----------
    conv : nn.Conv1d
        The 1-D convolution to run first.
    relu : nn.ReLU
        The rectifier applied to the convolution output.

    Notes
    -----
    This is a structural tag only — it carries no quantization state; observers are
    attached later by :func:`lucid.quantization.prepare_qat`.
    """

    def __init__(self, conv: nn.Conv1d, relu: nn.ReLU) -> None:
        super().__init__(conv, relu)


class ConvReLU2d(_FusedModule):
    """Fused float ``Conv2d`` + ``ReLU`` marker (a :class:`~lucid.nn.Sequential`).

    A thin :class:`~lucid.nn.Sequential` of ``[conv, relu]`` (with any BatchNorm already
    folded into the conv weight) emitted by :func:`lucid.quantization.fuse_modules` so
    that a later activation observer sees the *post*-ReLU range, letting the eventual
    quantized layer pick an output grid for the true (non-negative) inference range.  It
    runs as plain float at this stage; :func:`lucid.quantization.convert` maps it to a
    fused quantized :class:`~lucid.nn.quantized.ConvReLU2d`.

    Parameters
    ----------
    conv : nn.Conv2d
        The 2-D convolution to run first (BatchNorm already folded in).
    relu : nn.ReLU
        The rectifier applied to the convolution output.

    Notes
    -----
    This is a structural tag only — it carries no quantization state; observers are
    attached later by :func:`lucid.quantization.prepare_qat`.
    """

    def __init__(self, conv: nn.Conv2d, relu: nn.ReLU) -> None:
        super().__init__(conv, relu)


class ConvReLU3d(_FusedModule):
    """Fused float ``Conv3d`` + ``ReLU`` marker (a :class:`~lucid.nn.Sequential`).

    A thin :class:`~lucid.nn.Sequential` of ``[conv, relu]`` emitted by
    :func:`lucid.quantization.fuse_modules` so that a later activation observer sees the
    *post*-ReLU range, letting the eventual quantized layer pick an output grid for the
    true (non-negative) inference range.  It runs as plain float at this stage;
    :func:`lucid.quantization.convert` maps it to a fused quantized
    :class:`~lucid.nn.quantized.ConvReLU3d`.

    Parameters
    ----------
    conv : nn.Conv3d
        The 3-D convolution to run first.
    relu : nn.ReLU
        The rectifier applied to the convolution output.

    Notes
    -----
    This is a structural tag only — it carries no quantization state; observers are
    attached later by :func:`lucid.quantization.prepare_qat`.
    """

    def __init__(self, conv: nn.Conv3d, relu: nn.ReLU) -> None:
        super().__init__(conv, relu)


class LinearReLU(_FusedModule):
    """Fused float ``Linear`` + ``ReLU`` marker (a :class:`~lucid.nn.Sequential`).

    A thin :class:`~lucid.nn.Sequential` of ``[linear, relu]`` emitted by
    :func:`lucid.quantization.fuse_modules` so that a later activation observer sees the
    *post*-ReLU range, letting the eventual quantized layer pick an output grid for the
    true (non-negative) inference range.  It runs as plain float at this stage;
    :func:`lucid.quantization.convert` maps it to a fused quantized
    :class:`~lucid.nn.quantized.LinearReLU`.

    Parameters
    ----------
    linear : nn.Linear
        The linear (fully-connected) layer to run first.
    relu : nn.ReLU
        The rectifier applied to the linear output.

    Notes
    -----
    This is a structural tag only — it carries no quantization state; observers are
    attached later by :func:`lucid.quantization.prepare_qat`.
    """

    def __init__(self, linear: nn.Linear, relu: nn.ReLU) -> None:
        super().__init__(linear, relu)
