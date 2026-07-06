"""Quantized fused (intrinsic) modules — ``ConvReLU`` / ``LinearReLU``.

These are the quantized counterparts of the ``lucid.nn.intrinsic`` float
modules.  Each reuses the plain quantized layer and only overrides the
post-op activation hook to apply ReLU *before* the output is fake-quantized
— matching the calibration, where the fused module's activation observer
saw the post-ReLU range.  ``from_float`` delegates to the base builder via
``super()`` (so ``cls`` stays the fused subclass) after copying the fused
module's qconfig + observer onto its inner weighted child.
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized.conv import Conv1d, Conv2d, Conv3d
from lucid.nn.quantized.linear import Linear

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _wire_inner(mod: nn.Module) -> nn.Module:
    """Return the weighted child carrying the qparams.

    A **float** fused module is an ``nn.Sequential`` — copy its qconfig +
    observer onto ``seq[0]`` and return that.  A **QAT** fused module
    (``nn.qat.ConvReLU`` / ``LinearReLU``) is already a single weighted layer
    carrying ``weight_fake_quant`` + ``activation_post_process`` directly, so it
    is returned unchanged.
    """
    if not isinstance(mod, nn.Sequential):
        return mod
    inner = mod[0]
    inner.qconfig = mod.qconfig
    inner.activation_post_process = mod.activation_post_process
    return inner


class LinearReLU(Linear):
    r"""Quantized linear layer with a fused ReLU — int8 weight, float compute.

    A fused (intrinsic) block: the quantized counterpart of the float
    ``nn.intrinsic.LinearReLU``, installed by :func:`lucid.quantization.convert`
    / :meth:`from_float`. It behaves exactly like
    :class:`~lucid.nn.quantized.Linear` — int8 per-output-channel weight, float
    matmul, output fake-quantized to the calibrated grid — but folds a ReLU into
    the layer so the linear and its activation share a single quantized output.

    **Why fuse.** During calibration the fused module's activation observer sat
    *after* the ReLU, so it recorded the post-ReLU (non-negative) range. To
    reproduce those numerics at inference the ReLU must be applied *before* the
    output fake-quant — exactly what the overridden ``_activation`` hook does.
    Fusing also requantizes the whole ``linear → relu`` pair once, against the
    tighter non-negative range, which uses the int8 grid more efficiently than
    quantizing the raw linear output and clamping afterwards.

    Encoding (int8 weight) happens once at :meth:`from_float`; decode + matmul +
    fused ReLU + requantize run on every forward:

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{relu}(\hat{w}\, x^{\top} + b)\bigr),
        \qquad
        \hat{w}_{ij} = (w^{q}_{ij} - z_i)\, s_i

    where :math:`s_i, z_i` are the per-output-channel weight scale / zero-point,
    :math:`b` the float bias, and the outer fake-quant snaps the post-ReLU result
    to the scalar calibrated output ``(scale, zero_point)``.

    Parameters
    ----------
    in_features : int
        Size of each input sample — the ``in`` dimension of the weight.
    out_features : int
        Size of each output sample — the ``out`` dimension, quantized per-channel.
    bias : bool, optional
        Whether a (float) bias term is added *before* the fused ReLU. Defaults to
        ``True``.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` weight codes, shape ``(out_features, in_features)``.
    weight_scale : Tensor
        Per-output-channel weight scale, shape ``(out_features,)``.
    weight_zero_point : Tensor
        Per-output-channel weight zero-point, shape ``(out_features,)``.
    scale : Tensor
        Scalar output-activation scale (post-ReLU), from the calibrated observer.
    zero_point : Tensor
        Scalar output-activation zero-point, from the calibrated observer.
    bias : Tensor or None
        The float bias of shape ``(out_features,)``, or ``None``.

    Notes
    -----
    - Produced by ``convert`` / :meth:`from_float`, **not** constructed directly;
      the constructor is inherited unchanged from
      :class:`~lucid.nn.quantized.Linear` and only the ``_activation`` hook differs.
    - :meth:`from_float` accepts either a **float** fused module (an
      ``nn.Sequential`` whose ``[0]`` is the weighted layer) or a single-layer
      **QAT** fused module; the shared wiring copies the fused qconfig / observer
      onto the weighted child before quantizing.
    - The output range is non-negative because the ReLU precedes the fake-quant —
      calibrate the *fused* module (not the bare linear) so the observed grid
      matches what runs at inference.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> qlr = nn.quantized.LinearReLU(64, 32)     # normally built via convert()
    >>> qlr(lucid.randn(4, 64)).shape
    (4, 32)

    A directly constructed instance carries zeroed int8 weights and identity
    qparams — like :class:`Linear`, it returns garbage until ``from_float`` /
    ``load_state_dict`` populates the buffers:

    >>> bool((qlr.weight_int8 == 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.Linear : The un-fused quantized linear it extends.
    lucid.nn.quantized.ConvReLU2d : The convolutional fused analogue.
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    @override
    def _activation(self, y: Tensor) -> Tensor:
        return F.relu(y)

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> LinearReLU:
        return cast("LinearReLU", super().from_float(_wire_inner(mod)))


class ConvReLU1d(Conv1d):
    r"""Quantized 1-D convolution with a fused ReLU — int8 kernel, float compute.

    A fused (intrinsic) block: the quantized counterpart of the float
    ``nn.intrinsic.ConvReLU1d``, installed by :func:`lucid.quantization.convert`
    / :meth:`from_float`. It behaves exactly like
    :class:`~lucid.nn.quantized.Conv1d` — int8 per-output-channel kernel, float
    convolution, output fake-quantized to the calibrated grid — but folds a ReLU
    into the layer so the conv and its activation share one quantized output.

    **Why fuse.** During calibration the fused module's activation observer sat
    *after* the ReLU, so it recorded the post-ReLU (non-negative) range. To
    reproduce those numerics at inference the ReLU must run *before* the output
    fake-quant — the job of the overridden ``_activation`` hook. Requantizing the
    ``conv → relu`` pair once, against the tighter non-negative range, uses the
    int8 grid more efficiently than quantizing the raw conv output then clamping.

    Encoding (int8 kernel) happens once at :meth:`from_float`; decode + convolve +
    fused ReLU + requantize run on every forward:

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{relu}(\operatorname{conv1d}(x, \hat{w}) + b)\bigr),
        \qquad
        \hat{w}_{o,c,k} = (w^{q}_{o,c,k} - z_o)\, s_o

    where :math:`s_o, z_o` are the per-output-channel kernel scale / zero-point
    and the outer fake-quant snaps the post-ReLU result to the scalar calibrated
    output ``(scale, zero_point)``.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input signal.
    out_channels : int
        Number of channels produced by the convolution — quantized per-channel.
    kernel_size : int or tuple of int
        Size of the convolving kernel.
    stride : int or tuple of int
        Stride of the convolution.
    padding : int or tuple of int
        Implicit zero-padding added to both sides of the spatial axis.
    dilation : int or tuple of int
        Spacing between kernel elements.
    groups : int
        Number of blocked connections from input to output channels.
    bias : bool
        Whether a (float) bias term is added *before* the fused ReLU.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` kernel codes, shape ``(out_channels, in_channels/groups, k)``.
    weight_scale : Tensor
        Per-output-channel kernel scale, shape ``(out_channels,)``.
    weight_zero_point : Tensor
        Per-output-channel kernel zero-point, shape ``(out_channels,)``.
    scale, zero_point : Tensor
        Scalar output-activation qparams (post-ReLU), from the calibrated observer.
    bias : Tensor or None
        The float bias of shape ``(out_channels,)``, or ``None``.

    Notes
    -----
    - Produced by ``convert`` / :meth:`from_float`, **not** constructed directly;
      the constructor is inherited unchanged from
      :class:`~lucid.nn.quantized.Conv1d` and only the ``_activation`` hook differs.
    - :meth:`from_float` accepts a **float** fused ``nn.Sequential`` (``[0]`` is
      the conv) or a single-layer **QAT** fused module; the shared wiring copies
      the fused qconfig / observer onto the weighted child before quantizing.
    - Output is non-negative (ReLU precedes the fake-quant) — calibrate the
      *fused* module so the observed grid matches inference.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> q = nn.quantized.ConvReLU1d(4, 8, 3, 1, 1, 1, 1, True)   # via convert()
    >>> q(lucid.randn(1, 4, 16)).shape
    (1, 8, 16)
    >>> bool((q.weight_int8 == 0).all().item())      # zeroed until from_float
    True

    See Also
    --------
    lucid.nn.quantized.Conv1d : The un-fused quantized 1-D conv it extends.
    lucid.nn.quantized.ConvReLU2d : The 2-D fused analogue (vision workhorse).
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    @override
    def _activation(self, y: Tensor) -> Tensor:
        return F.relu(y)

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> ConvReLU1d:
        return cast("ConvReLU1d", super().from_float(_wire_inner(mod)))


class ConvReLU2d(Conv2d):
    r"""Quantized 2-D convolution with a fused ReLU — int8 kernel, float compute.

    The most common fused block in the quantized vision zoo: nearly every
    ``conv → bn → relu`` triple in a ResNet-class backbone folds down to a
    ``ConvReLU2d`` after batch-norm folding. It is the quantized counterpart of
    the float ``nn.intrinsic.ConvReLU2d``, installed by
    :func:`lucid.quantization.convert` / :meth:`from_float`, and behaves exactly
    like :class:`~lucid.nn.quantized.Conv2d` — int8 per-output-channel kernel,
    float convolution, output fake-quantized to the calibrated grid — but folds a
    ReLU into the layer so conv + activation share one quantized output.

    **Why fuse.** During calibration the fused module's activation observer sat
    *after* the ReLU, so it recorded the post-ReLU (non-negative) range. To
    reproduce those numerics at inference the ReLU must run *before* the output
    fake-quant — the job of the overridden ``_activation`` hook. Requantizing the
    ``conv → relu`` pair once, against the tighter non-negative range, uses the
    int8 grid more efficiently than quantizing the raw conv output then clamping.

    Encoding (int8 kernel) happens once at :meth:`from_float`; decode + convolve +
    fused ReLU + requantize run on every forward:

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{relu}(\operatorname{conv2d}(x, \hat{w}) + b)\bigr),
        \qquad
        \hat{w}_{o,c,i,j} = (w^{q}_{o,c,i,j} - z_o)\, s_o

    where :math:`s_o, z_o` are the per-output-channel kernel scale / zero-point
    and the outer fake-quant snaps the post-ReLU result to the scalar calibrated
    output ``(scale, zero_point)``.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution — quantized per-channel.
    kernel_size : int or tuple of int
        Size of the convolving kernel.
    stride : int or tuple of int
        Stride of the convolution.
    padding : int or tuple of int
        Implicit zero-padding added to both sides of each spatial dim.
    dilation : int or tuple of int
        Spacing between kernel elements.
    groups : int
        Number of blocked connections from input to output channels.
    bias : bool
        Whether a (float) bias term is added *before* the fused ReLU.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` kernel codes, shape
        ``(out_channels, in_channels/groups, kh, kw)``.
    weight_scale : Tensor
        Per-output-channel kernel scale, shape ``(out_channels,)``.
    weight_zero_point : Tensor
        Per-output-channel kernel zero-point, shape ``(out_channels,)``.
    scale, zero_point : Tensor
        Scalar output-activation qparams (post-ReLU), from the calibrated observer.
    bias : Tensor or None
        The float bias of shape ``(out_channels,)``, or ``None``.

    Notes
    -----
    - Produced by ``convert`` / :meth:`from_float`, **not** constructed directly;
      the constructor is inherited unchanged from
      :class:`~lucid.nn.quantized.Conv2d` and only the ``_activation`` hook differs.
    - :meth:`from_float` accepts a **float** fused ``nn.Sequential`` (``[0]`` is
      the conv) or a single-layer **QAT** fused module; the shared wiring copies
      the fused qconfig / observer onto the weighted child before quantizing.
    - Output is non-negative (ReLU precedes the fake-quant) — calibrate the
      *fused* module so the observed grid matches inference.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> q = nn.quantized.ConvReLU2d(3, 8, 3, 1, 1, 1, 1, True)   # via convert()
    >>> q(lucid.randn(1, 3, 8, 8)).shape
    (1, 8, 8, 8)
    >>> bool((q.weight_int8 == 0).all().item())      # zeroed until from_float
    True

    See Also
    --------
    lucid.nn.quantized.Conv2d : The un-fused quantized 2-D conv it extends.
    lucid.nn.quantized.ConvReLU1d : The 1-D fused analogue.
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    @override
    def _activation(self, y: Tensor) -> Tensor:
        return F.relu(y)

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> ConvReLU2d:
        return cast("ConvReLU2d", super().from_float(_wire_inner(mod)))


class ConvReLU3d(Conv3d):
    r"""Quantized 3-D convolution with a fused ReLU — int8 kernel, float compute.

    A fused (intrinsic) block for volumetric / video models: the quantized
    counterpart of the float ``nn.intrinsic.ConvReLU3d``, installed by
    :func:`lucid.quantization.convert` / :meth:`from_float`. It behaves exactly
    like :class:`~lucid.nn.quantized.Conv3d` — int8 per-output-channel kernel,
    float convolution, output fake-quantized to the calibrated grid — but folds a
    ReLU into the layer so conv + activation share one quantized output. The 3-D
    kernel makes int8 storage especially valuable here, since a volumetric filter
    ``(out, in/groups, kd, kh, kw)`` is large.

    **Why fuse.** During calibration the fused module's activation observer sat
    *after* the ReLU, so it recorded the post-ReLU (non-negative) range. To
    reproduce those numerics at inference the ReLU must run *before* the output
    fake-quant — the job of the overridden ``_activation`` hook. Requantizing the
    ``conv → relu`` pair once, against the tighter non-negative range, uses the
    int8 grid more efficiently than quantizing the raw conv output then clamping.

    Encoding (int8 kernel) happens once at :meth:`from_float`; decode + convolve +
    fused ReLU + requantize run on every forward:

    .. math::

        y = \operatorname{fake\_quant}\!\bigl(
            \operatorname{relu}(\operatorname{conv3d}(x, \hat{w}) + b)\bigr),
        \qquad
        \hat{w}_{o,c,d,i,j} = (w^{q}_{o,c,d,i,j} - z_o)\, s_o

    where :math:`s_o, z_o` are the per-output-channel kernel scale / zero-point
    and the outer fake-quant snaps the post-ReLU result to the scalar calibrated
    output ``(scale, zero_point)``.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input volume.
    out_channels : int
        Number of channels produced by the convolution — quantized per-channel.
    kernel_size : int or tuple of int
        Size of the convolving kernel.
    stride : int or tuple of int
        Stride of the convolution.
    padding : int or tuple of int
        Implicit zero-padding added to both sides of each spatial dim.
    dilation : int or tuple of int
        Spacing between kernel elements.
    groups : int
        Number of blocked connections from input to output channels.
    bias : bool
        Whether a (float) bias term is added *before* the fused ReLU.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` kernel codes, shape
        ``(out_channels, in_channels/groups, kd, kh, kw)``.
    weight_scale : Tensor
        Per-output-channel kernel scale, shape ``(out_channels,)``.
    weight_zero_point : Tensor
        Per-output-channel kernel zero-point, shape ``(out_channels,)``.
    scale, zero_point : Tensor
        Scalar output-activation qparams (post-ReLU), from the calibrated observer.
    bias : Tensor or None
        The float bias of shape ``(out_channels,)``, or ``None``.

    Notes
    -----
    - Produced by ``convert`` / :meth:`from_float`, **not** constructed directly;
      the constructor is inherited unchanged from
      :class:`~lucid.nn.quantized.Conv3d` and only the ``_activation`` hook differs.
    - :meth:`from_float` accepts a **float** fused ``nn.Sequential`` (``[0]`` is
      the conv) or a single-layer **QAT** fused module; the shared wiring copies
      the fused qconfig / observer onto the weighted child before quantizing.
    - Output is non-negative (ReLU precedes the fake-quant) — calibrate the
      *fused* module so the observed grid matches inference.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> q = nn.quantized.ConvReLU3d(2, 4, 3, 1, 1, 1, 1, True)   # via convert()
    >>> q(lucid.randn(1, 2, 8, 8, 8)).shape
    (1, 4, 8, 8, 8)
    >>> bool((q.weight_int8 == 0).all().item())      # zeroed until from_float
    True

    See Also
    --------
    lucid.nn.quantized.Conv3d : The un-fused quantized 3-D conv it extends.
    lucid.nn.quantized.ConvReLU2d : The 2-D fused analogue (vision workhorse).
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    @override
    def _activation(self, y: Tensor) -> Tensor:
        return F.relu(y)

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> ConvReLU3d:
        return cast("ConvReLU3d", super().from_float(_wire_inner(mod)))
