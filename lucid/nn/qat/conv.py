"""QAT ``Conv1d`` / ``Conv2d`` / ``Conv3d`` — weight + output fake-quant.

Same idea as the QAT :class:`~lucid.nn.qat.Linear`: a trainable float kernel
with fake-quant on the weight and the output, so gradients (via STE) adapt
the weights to the eventual int8 grid.  Integer / tuple padding is supported
(string ``"same"`` / ``"valid"`` is deferred, as in the quantized conv).
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig


def _conv_from_float(cls: type, mod: nn.Module) -> nn.Module:
    """Build a QAT conv from a float conv, sharing its trained kernel."""
    c = cast("nn.Conv2d", mod)  # structural: all conv ranks share these attrs
    qat = cast(
        "nn.Conv2d",
        cls(
            c.in_channels,
            c.out_channels,
            c.kernel_size,
            stride=c.stride,
            padding=c.padding,
            dilation=c.dilation,
            groups=c.groups,
            bias=c.bias is not None,
            qconfig=cast("QConfig", mod.qconfig),  # set by prepare_qat
        ),
    )
    # Adopt the trained kernel/bias directly — the prepare_qat deep-copy
    # already produced independent Parameters for this module tree.
    qat.weight = c.weight
    if c.bias is not None:
        qat.bias = c.bias
    return qat


class Conv1d(nn.Conv1d):
    """Quantization-aware 1-D convolution (trainable float kernel, fake-quant).

    A trainable float :class:`~lucid.nn.Conv1d` that fake-quantizes both the weight
    and the output on every forward through a straight-through estimator (STE), so the
    network adapts its weights to the eventual int8 grid while still training in full
    precision.  Inserted in place of a float conv by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` later
    bakes the trained kernel and the observed qparams into an inference
    :class:`~lucid.nn.quantized.Conv1d`.

    Parameters
    ----------
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
    *args, **kwargs
        Remaining arguments are forwarded verbatim to the float
        :class:`~lucid.nn.Conv1d` constructor (``in_channels``, ``out_channels``,
        ``kernel_size``, ``stride``, ``padding``, ``dilation``, ``groups``, ``bias``).

    Notes
    -----
    The weight fake-quant rounds the kernel while the STE passes gradients straight
    through, keeping the layer fully differentiable.  String padding (``"same"`` /
    ``"valid"``) is deferred, matching the quantized conv.
    """

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(
        self, *args: object, qconfig: QConfig | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # forwarded to nn.Conv1d
        if qconfig is None:
            raise ValueError("qat conv requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary conv layer
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.conv1d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return cast("Tensor", self.activation_post_process(y))

    @classmethod
    def from_float(cls, mod: nn.Module) -> Conv1d:
        return cast("Conv1d", _conv_from_float(cls, mod))


class Conv2d(nn.Conv2d):
    """Quantization-aware 2-D convolution (trainable float kernel, fake-quant).

    A trainable float :class:`~lucid.nn.Conv2d` that fake-quantizes both the weight
    and the output on every forward through a straight-through estimator (STE), so the
    network adapts its weights to the eventual int8 grid while still training in full
    precision.  Inserted in place of a float conv by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` later
    bakes the trained kernel and the observed qparams into an inference
    :class:`~lucid.nn.quantized.Conv2d`.

    Parameters
    ----------
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
    *args, **kwargs
        Remaining arguments are forwarded verbatim to the float
        :class:`~lucid.nn.Conv2d` constructor (``in_channels``, ``out_channels``,
        ``kernel_size``, ``stride``, ``padding``, ``dilation``, ``groups``, ``bias``).

    Notes
    -----
    The weight fake-quant rounds the kernel while the STE passes gradients straight
    through, keeping the layer fully differentiable.  String padding (``"same"`` /
    ``"valid"``) is deferred, matching the quantized conv.
    """

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(
        self, *args: object, qconfig: QConfig | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # forwarded to nn.Conv2d
        if qconfig is None:
            raise ValueError("qat conv requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary conv layer
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.conv2d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return cast("Tensor", self.activation_post_process(y))

    @classmethod
    def from_float(cls, mod: nn.Module) -> Conv2d:
        return cast("Conv2d", _conv_from_float(cls, mod))


class Conv3d(nn.Conv3d):
    """Quantization-aware 3-D convolution (trainable float kernel, fake-quant).

    A trainable float :class:`~lucid.nn.Conv3d` that fake-quantizes both the weight
    and the output on every forward through a straight-through estimator (STE), so the
    network adapts its weights to the eventual int8 grid while still training in full
    precision.  Inserted in place of a float conv by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` later
    bakes the trained kernel and the observed qparams into an inference
    :class:`~lucid.nn.quantized.Conv3d`.

    Parameters
    ----------
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
    *args, **kwargs
        Remaining arguments are forwarded verbatim to the float
        :class:`~lucid.nn.Conv3d` constructor (``in_channels``, ``out_channels``,
        ``kernel_size``, ``stride``, ``padding``, ``dilation``, ``groups``, ``bias``).

    Notes
    -----
    The weight fake-quant rounds the kernel while the STE passes gradients straight
    through, keeping the layer fully differentiable.  String padding (``"same"`` /
    ``"valid"``) is deferred, matching the quantized conv.
    """

    weight_fake_quant: FakeQuantize
    activation_post_process: FakeQuantize

    def __init__(
        self, *args: object, qconfig: QConfig | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]  # forwarded to nn.Conv3d
        if qconfig is None:
            raise ValueError("qat conv requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())
        self.activation_post_process = cast("FakeQuantize", qconfig.activation())

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary conv layer
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.conv3d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return cast("Tensor", self.activation_post_process(y))

    @classmethod
    def from_float(cls, mod: nn.Module) -> Conv3d:
        return cast("Conv3d", _conv_from_float(cls, mod))


def _fused_conv_from_float(cls: type, mod: nn.Module) -> nn.Module:
    """Build a QAT conv-relu from a fused float ``nni.ConvReLU`` (its inner conv)."""
    inner = cast("nn.Sequential", mod)[0]
    inner.qconfig = mod.qconfig
    return _conv_from_float(cls, inner)


class ConvReLU1d(Conv1d):
    """Quantization-aware fused 1-D convolution + ReLU (trainable, fake-quant).

    Behaves like :class:`Conv1d`, but the activation fake-quant observes the range
    *after* the fused ReLU, so the calibrated output grid reflects the true
    (non-negative) inference range.  Built from a fused float
    :class:`~lucid.nn.intrinsic.ConvReLU1d` by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` folds it
    into a single quantized :class:`~lucid.nn.quantized.ConvReLU1d`.

    Parameters
    ----------
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
    *args, **kwargs
        Remaining arguments are forwarded to the float :class:`~lucid.nn.Conv1d`
        constructor, exactly as for :class:`Conv1d`.

    Notes
    -----
    The weight and the post-ReLU output are both fake-quantized every forward via a
    straight-through estimator, so gradients reach the float kernel unchanged.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary conv layer
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.conv1d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return cast("Tensor", self.activation_post_process(F.relu(y)))

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> "ConvReLU1d":
        return cast("ConvReLU1d", _fused_conv_from_float(cls, mod))


class ConvReLU2d(Conv2d):
    """Quantization-aware fused 2-D convolution + ReLU (trainable, fake-quant).

    Behaves like :class:`Conv2d`, but the activation fake-quant observes the range
    *after* the fused ReLU, so the calibrated output grid reflects the true
    (non-negative) inference range.  Built from a fused float
    :class:`~lucid.nn.intrinsic.ConvReLU2d` by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` folds it
    into a single quantized :class:`~lucid.nn.quantized.ConvReLU2d`.

    Parameters
    ----------
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
    *args, **kwargs
        Remaining arguments are forwarded to the float :class:`~lucid.nn.Conv2d`
        constructor, exactly as for :class:`Conv2d`.

    Notes
    -----
    The weight and the post-ReLU output are both fake-quantized every forward via a
    straight-through estimator, so gradients reach the float kernel unchanged.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary conv layer
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.conv2d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return cast("Tensor", self.activation_post_process(F.relu(y)))

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> "ConvReLU2d":
        return cast("ConvReLU2d", _fused_conv_from_float(cls, mod))


class ConvReLU3d(Conv3d):
    """Quantization-aware fused 3-D convolution + ReLU (trainable, fake-quant).

    Behaves like :class:`Conv3d`, but the activation fake-quant observes the range
    *after* the fused ReLU, so the calibrated output grid reflects the true
    (non-negative) inference range.  Built from a fused float
    :class:`~lucid.nn.intrinsic.ConvReLU3d` by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` folds it
    into a single quantized :class:`~lucid.nn.quantized.ConvReLU3d`.

    Parameters
    ----------
    qconfig : QConfig
        Quantization recipe supplying the weight and activation
        :class:`~lucid.quantization.FakeQuantize` modules applied during training.
    *args, **kwargs
        Remaining arguments are forwarded to the float :class:`~lucid.nn.Conv3d`
        constructor, exactly as for :class:`Conv3d`.

    Notes
    -----
    The weight and the post-ReLU output are both fake-quantized every forward via a
    straight-through estimator, so gradients reach the float kernel unchanged.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # unary conv layer
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        y = F.conv3d(
            x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return cast("Tensor", self.activation_post_process(F.relu(y)))

    @classmethod
    @override
    def from_float(cls, mod: nn.Module) -> "ConvReLU3d":
        return cast("ConvReLU3d", _fused_conv_from_float(cls, mod))
