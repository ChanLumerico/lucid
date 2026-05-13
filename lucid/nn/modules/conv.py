"""
Convolution and transposed convolution modules.
"""

import math
from typing import Callable

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike, _Size2d, _Size3d
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init
from lucid.nn.functional.conv import (
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
)
from lucid.nn.functional.sampling import pad as _F_pad

_VALID_PADDING_MODES = frozenset({"zeros", "reflect", "replicate", "circular"})

# Maps Conv padding_mode to F.pad mode.
_PADDING_MODE_TO_FPAD = {
    "zeros": "constant",
    "reflect": "reflect",
    "replicate": "replicate",
    "circular": "circular",
}


def _pair(v: _Size2d) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]


def _triple(v: _Size3d) -> tuple[int, int, int]:
    return (v, v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]


def _same_pad_pair(
    in_size: int, kernel: int, stride: int, dilation: int
) -> tuple[int, int]:
    """Compute (pad_lo, pad_hi) for `padding="same"` on one spatial dim.

    Reference parity: pad_lo = pad_total // 2, pad_hi = pad_total - pad_lo.
    For odd pad_total this is asymmetric (more padding on the high side).
    """
    out_size = (in_size + stride - 1) // stride
    pad_total = max(0, (out_size - 1) * stride + (kernel - 1) * dilation + 1 - in_size)
    pad_lo = pad_total // 2
    return pad_lo, pad_total - pad_lo


def _check_same_supported(stride_tuple: tuple[int, ...]) -> None:
    if any(s != 1 for s in stride_tuple):
        raise ValueError(
            "padding='same' is not supported with stride > 1 "
            f"(got stride={stride_tuple})"
        )


def _validate_padding_mode(mode: str) -> str:
    if mode not in _VALID_PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {sorted(_VALID_PADDING_MODES)}, got {mode!r}"
        )
    return mode


def _validate_int_padding(padding: object, label: str) -> None:
    """Reject string padding for ConvTranspose."""
    if isinstance(padding, str):
        raise ValueError(
            f"{label}: string padding ({padding!r}) is not supported; "
            "use an int or tuple of ints"
        )


_ConvFn = Callable[..., Tensor]


def _conv_forward_with_mode(
    x: Tensor,
    weight: Parameter,
    bias: Parameter | None,
    stride: tuple[int, ...],
    pad_lo: tuple[int, ...],
    pad_hi: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    padding_mode: str,
    conv_fn: _ConvFn,
) -> Tensor:
    """Dispatch a forward conv with arbitrary padding_mode and asymmetric pad.

    `pad_lo` / `pad_hi` are per-spatial-dim (first→last) padding amounts.
    When `padding_mode == "zeros"` and pad is symmetric, the engine conv
    handles the padding directly.  Otherwise we pre-pad via `F.pad` and
    call conv with padding=0 along that axis.
    """
    n: int = len(stride)
    symmetric: bool = all(pad_lo[i] == pad_hi[i] for i in range(n))
    if padding_mode == "zeros" and symmetric:
        engine_pad: int | tuple[int, ...] = pad_lo[0] if n == 1 else tuple(pad_lo)
        return _call_conv(
            conv_fn, x, weight, bias, stride, engine_pad, dilation, groups, n
        )
    # Pre-pad path.  F.pad uses last-dim-first flat tuple.
    # pad_lo[i], pad_hi[i] are spatial dim i (first→last); we need to reverse
    # so that the LAST spatial dim comes first in F.pad's flat tuple.
    pad_flat: list[int] = []
    for i in reversed(range(n)):
        pad_flat.extend([pad_lo[i], pad_hi[i]])
    fpad_mode: str = _PADDING_MODE_TO_FPAD[padding_mode]
    x_padded: Tensor = _F_pad(x, tuple(pad_flat), mode=fpad_mode)
    zero_pad: int | tuple[int, ...] = 0 if n == 1 else (0,) * n
    return _call_conv(
        conv_fn, x_padded, weight, bias, stride, zero_pad, dilation, groups, n
    )


def _call_conv(
    conv_fn: _ConvFn,
    x: Tensor,
    weight: Parameter,
    bias: Parameter | None,
    stride: tuple[int, ...],
    padding: int | tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    n: int,
) -> Tensor:
    """Invoke conv_fn with the right argument arity.

    conv1d/2d/3d in `nn.functional.conv` accept (x, weight, bias, stride, padding,
    dilation, groups) but conv1d takes scalar ints while conv2d/3d take tuples.
    """
    if n == 1:
        s: int = stride[0] if isinstance(stride, tuple) else stride
        d: int = dilation[0] if isinstance(dilation, tuple) else dilation
        p: int = padding[0] if isinstance(padding, tuple) else padding
        return conv_fn(x, weight, bias, s, p, d, groups)
    return conv_fn(x, weight, bias, stride, padding, dilation, groups)


class Conv1d(Module):
    r"""Applies a 1D convolution over a sequence of input signals.

    A 1D convolution slides a learned filter (kernel) across a 1D sequence
    and computes a dot product at each position.  Strictly speaking the
    operation is *cross-correlation*:

    .. math::

        y[n, c_{\text{out}}, l] = \sum_{c_{\text{in}}=0}^{C_{\text{in}}/g - 1}
            \sum_{k=0}^{K-1}
            x\!\left[n,\, c_{\text{in}},\, l \cdot s + k \cdot d\right]
            \cdot W\!\left[c_{\text{out}},\, c_{\text{in}},\, k\right]
            + b\!\left[c_{\text{out}}\right]

    where :math:`s` is the stride, :math:`d` is the dilation factor,
    :math:`g` is the number of groups, and :math:`K` is the kernel size.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input signal.
    out_channels : int
        Number of channels produced by the convolution (i.e. number of
        independent filters).
    kernel_size : int
        Length of the 1D convolving kernel.
    stride : int, optional
        Step size between consecutive kernel placements along the length
        dimension. Default: ``1``.
    padding : int or str, optional
        Zero-padding added to both sides of the input before the
        convolution.  Can also be ``"same"`` (output length equals
        ``ceil(L_in / stride)``, requires ``stride=1``) or ``"valid"``
        (no padding; identical to ``padding=0``). Default: ``0``.
    dilation : int, optional
        Spacing between kernel elements.  A dilation of ``d`` inserts
        ``d - 1`` zeros between consecutive kernel weights, expanding the
        effective receptive field without increasing parameter count
        (also called *atrous* convolution). Default: ``1``.
    groups : int, optional
        Splits the input and output channels into ``groups`` independent
        paths.  ``in_channels`` and ``out_channels`` must both be
        divisible by ``groups``.  Setting ``groups = in_channels``
        gives *depthwise* convolution. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias of shape ``(out_channels,)`` is
        added to the output. Default: ``True``.
    padding_mode : str, optional
        Padding strategy when ``padding`` is an integer.  One of
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable filter tensor of shape
        ``(out_channels, in_channels // groups, kernel_size)``.
        Initialized with Kaiming uniform:

        .. math::

            \text{fan\_in} = \frac{C_{\text{in}}}{g} \cdot K, \quad
            W \sim \mathcal{U}\!\left[
                -\sqrt{\tfrac{6}{\text{fan\_in}}},\;
                \sqrt{\tfrac{6}{\text{fan\_in}}}
            \right]

        (using :math:`a = \sqrt{5}` in the Kaiming formula).
    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None`` if
        ``bias=False``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, L)`
    Output:
        :math:`(N, C_{\text{out}}, L_{\text{out}})` where

        .. math::

            L_{\text{out}} = \left\lfloor
                \frac{L + 2p - d(K - 1) - 1}{s} + 1
            \right\rfloor

        and :math:`p` is the (symmetric) padding, :math:`d` is dilation,
        :math:`K` is kernel size, :math:`s` is stride.

    Notes
    -----
    **Groups and depthwise convolution.**  When ``groups = g > 1``, the
    input channels are split into ``g`` groups of size
    ``in_channels // g``, each group is convolved independently, and the
    results are concatenated.  Depthwise convolution (``groups =
    in_channels``) applies one filter per input channel, dramatically
    reducing the parameter count compared to a full convolution.

    **Dilated (atrous) convolution.**  Setting ``dilation > 1`` inserts
    gaps between kernel taps, enlarging the receptive field without
    additional parameters or pooling.  This is commonly used in semantic
    segmentation and WaveNet-style audio models.

    **Padding modes.**  ``"reflect"`` and ``"circular"`` padding
    avoid boundary artefacts that zero-padding can introduce;
    ``"replicate"`` repeats edge values.  These modes are applied via
    a pre-padding step followed by a ``padding=0`` convolution.

    Examples
    --------
    Basic 1D convolution:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    >>> x = lucid.zeros(4, 1, 16)   # (N, C_in, L)
    >>> y = conv(x)
    >>> y.shape
    (4, 8, 16)

    Depthwise 1D convolution (one filter per channel):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> depthwise = nn.Conv1d(
    ...     in_channels=32, out_channels=32,
    ...     kernel_size=5, padding=2, groups=32
    ... )
    >>> x = lucid.zeros(2, 32, 64)
    >>> y = depthwise(x)
    >>> y.shape
    (2, 32, 64)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = _validate_padding_mode(padding_mode)
        if isinstance(padding, str):
            mode = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported((stride,))
            self._padding_str: str | None = mode
            self.padding: int = 0
        else:
            self._padding_str = None
            self.padding = padding
        self.weight = Parameter(
            empty(
                out_channels,
                in_channels // groups,
                kernel_size,
                dtype=dtype,
                device=device,
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _resolve_pad(self, x: Tensor) -> tuple[tuple[int], tuple[int]]:
        if self._padding_str == "valid":
            return (0,), (0,)
        if self._padding_str == "same":
            lo, hi = _same_pad_pair(
                x.shape[2], self.kernel_size, self.stride, self.dilation
            )
            return (lo,), (hi,)
        return (self.padding,), (self.padding,)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        pad_lo, pad_hi = self._resolve_pad(x)
        return _conv_forward_with_mode(
            x,
            self.weight,
            self.bias,
            (self.stride,),
            pad_lo,
            pad_hi,
            (self.dilation,),
            self.groups,
            self.padding_mode,
            conv1d,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class Conv2d(Module):
    r"""Applies a 2D convolution over a batch of images or feature maps.

    This module computes the 2D *cross-correlation* between the input
    and a set of learnable filters.  For a single output channel and a
    single input channel the operation is:

    .. math::

        y[n, c_{\text{out}}, h, w] =
        \sum_{c_{\text{in}}=0}^{C_{\text{in}}/g - 1}
        \sum_{k_h=0}^{K_H-1} \sum_{k_w=0}^{K_W-1}
        x\!\left[n,\; c_{\text{in}},\;
                  h \cdot s_h + k_h \cdot d_h,\;
                  w \cdot s_w + k_w \cdot d_w\right]
        \cdot W\!\left[c_{\text{out}},\; c_{\text{in}},\; k_h,\; k_w\right]
        + b\!\left[c_{\text{out}}\right]

    where :math:`(s_h, s_w)` is the stride, :math:`(d_h, d_w)` is the
    dilation factor, and :math:`g` is the number of groups.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple[int, int]
        Size of the convolving kernel.  A single ``int`` is broadcast
        to ``(kernel_size, kernel_size)``.
    stride : int or tuple[int, int], optional
        Stride of the convolution. Default: ``1``.
    padding : int, tuple[int, int], or str, optional
        Padding added to all four sides of the input.  ``"same"`` pads
        so the output spatial size equals ``ceil(H_in / s)`` (requires
        ``stride=1``); ``"valid"`` means no padding. Default: ``0``.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements (atrous / dilated convolution).
        Default: ``1``.
    groups : int, optional
        Number of blocked connections from input channels to output
        channels.  Both ``in_channels`` and ``out_channels`` must be
        divisible by ``groups``.  ``groups = in_channels`` gives
        *depthwise* convolution. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias to the output. Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable filter tensor of shape
        ``(out_channels, in_channels // groups, K_H, K_W)``.
        Initialized with Kaiming uniform using :math:`a = \sqrt{5}`:

        .. math::

            \text{fan\_in} = \frac{C_{\text{in}}}{g} \cdot K_H \cdot K_W, \quad
            W \sim \mathcal{U}\!\left[
                -\sqrt{\tfrac{6}{\text{fan\_in}}},\;
                \sqrt{\tfrac{6}{\text{fan\_in}}}
            \right]

    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, H, W)`
    Output:
        :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` where

        .. math::

            H_{\text{out}} = \left\lfloor
                \frac{H + 2p_h - d_h(K_H - 1) - 1}{s_h} + 1
            \right\rfloor, \quad
            W_{\text{out}} = \left\lfloor
                \frac{W + 2p_w - d_w(K_W - 1) - 1}{s_w} + 1
            \right\rfloor

    Notes
    -----
    **Groups and depthwise convolution.**  When ``groups = in_channels``
    each input channel is convolved with its own filter, yielding
    *depthwise* convolution.  This is the building block of
    MobileNet-style architectures.  A subsequent ``groups=1`` Conv2d
    with kernel size 1 (pointwise convolution) forms a
    *depthwise-separable* block.

    **Dilated (atrous) convolution.**  ``dilation > 1`` enlarges the
    receptive field of each kernel tap without increasing the number of
    parameters or reducing the spatial resolution.  Widely used in
    semantic segmentation (DeepLab) and generative models.

    **padding="same".**  Mimics the ``SAME`` padding convention: output
    is spatially identical in size to the input.  When the required
    total padding is odd, the extra pixel is added on the bottom/right
    side (low side gets ``pad_total // 2``).  Requires ``stride=1``.

    Examples
    --------
    Basic image convolution:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
    >>> x = lucid.zeros(8, 3, 32, 32)   # (N, C_in, H, W)
    >>> y = conv(x)
    >>> y.shape
    (8, 64, 32, 32)

    Depthwise separable convolution block:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> depthwise  = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
    >>> pointwise  = nn.Conv2d(32, 64, kernel_size=1)
    >>> x = lucid.zeros(4, 32, 16, 16)
    >>> y = pointwise(depthwise(x))
    >>> y.shape
    (4, 64, 16, 16)

    Dilated convolution (receptive field 9×9 with only 3×3 parameters):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> dilated = nn.Conv2d(1, 1, kernel_size=3, padding=4, dilation=4)
    >>> x = lucid.zeros(1, 1, 16, 16)
    >>> y = dilated(x)
    >>> y.shape
    (1, 1, 16, 16)

    Convolution without bias:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv_no_bias = nn.Conv2d(16, 32, kernel_size=1, bias=False)
    >>> x = lucid.zeros(2, 16, 8, 8)
    >>> y = conv_no_bias(x)
    >>> y.shape
    (2, 32, 8, 8)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d | str = 0,
        dilation: _Size2d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = _validate_padding_mode(padding_mode)
        if isinstance(padding, str):
            mode = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported(self.stride)
            self._padding_str: str | None = mode
            self.padding: tuple[int, int] = (0, 0)
        else:
            self._padding_str = None
            self.padding = _pair(padding)
        self.weight = Parameter(
            empty(
                out_channels, in_channels // groups, kh, kw, dtype=dtype, device=device
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _resolve_pad(self, x: Tensor) -> tuple[tuple[int, int], tuple[int, int]]:
        if self._padding_str == "valid":
            return (0, 0), (0, 0)
        if self._padding_str == "same":
            kh, kw = self.kernel_size
            sh, sw = self.stride
            dh, dw = self.dilation
            lo_h, hi_h = _same_pad_pair(x.shape[2], kh, sh, dh)
            lo_w, hi_w = _same_pad_pair(x.shape[3], kw, sw, dw)
            return (lo_h, lo_w), (hi_h, hi_w)
        return self.padding, self.padding

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        pad_lo, pad_hi = self._resolve_pad(x)
        return _conv_forward_with_mode(
            x,
            self.weight,
            self.bias,
            self.stride,
            pad_lo,
            pad_hi,
            self.dilation,
            self.groups,
            self.padding_mode,
            conv2d,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class Conv3d(Module):
    r"""Applies a 3D convolution over volumetric data (e.g. video or medical scans).

    Computes the 3D *cross-correlation* of the input with a bank of
    learnable 3D filters:

    .. math::

        y[n, c_{\text{out}}, d, h, w] =
        \sum_{c_{\text{in}}=0}^{C_{\text{in}}/g - 1}
        \sum_{k_d, k_h, k_w}
        x\!\left[n,\; c_{\text{in}},\;
                  d \cdot s_d + k_d \cdot d_d,\;
                  h \cdot s_h + k_h \cdot d_h,\;
                  w \cdot s_w + k_w \cdot d_w\right]
        \cdot W\!\left[c_{\text{out}},\; c_{\text{in}},\; k_d,\; k_h,\; k_w\right]
        + b\!\left[c_{\text{out}}\right]

    Parameters
    ----------
    in_channels : int
        Number of channels in the input volume.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple[int, int, int]
        Size of the 3D convolving kernel ``(K_D, K_H, K_W)``.
        A single ``int`` is broadcast to all three dimensions.
    stride : int or tuple[int, int, int], optional
        Stride along each spatial dimension. Default: ``1``.
    padding : int, tuple[int, int, int], or str, optional
        Zero-padding added on both sides along each spatial dimension.
        Accepts ``"same"`` (requires ``stride=1``) or ``"valid"``.
        Default: ``0``.
    dilation : int or tuple[int, int, int], optional
        Spacing between kernel elements. Default: ``1``.
    groups : int, optional
        Number of blocked connections.  ``groups = in_channels`` gives
        depthwise 3D convolution. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias. Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable filter tensor of shape
        ``(out_channels, in_channels // groups, K_D, K_H, K_W)``.
        Initialized with Kaiming uniform:

        .. math::

            \text{fan\_in} = \frac{C_{\text{in}}}{g}
                             \cdot K_D \cdot K_H \cdot K_W, \quad
            W \sim \mathcal{U}\!\left[
                -\sqrt{\tfrac{6}{\text{fan\_in}}},\;
                \sqrt{\tfrac{6}{\text{fan\_in}}}
            \right]

    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, D, H, W)`
    Output:
        :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})` where

        .. math::

            X_{\text{out}} = \left\lfloor
                \frac{X + 2p_x - d_x(K_X - 1) - 1}{s_x} + 1
            \right\rfloor \quad \text{for } X \in \{D, H, W\}

    Notes
    -----
    **Typical use cases.**  Conv3d is the standard building block for
    video understanding (3D ResNets, SlowFast), medical image analysis
    (CT/MRI volumetric segmentation), and point-cloud processing.  It is
    computationally heavier than Conv2d by a factor of roughly
    :math:`K_D` per layer; factorised (2+1)D convolutions are a common
    approximation.

    **Memory.**  A single 3D feature map can be large; consider
    ``groups > 1`` or smaller ``kernel_size`` when memory is a concern.

    Examples
    --------
    Basic volumetric convolution:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv3 = nn.Conv3d(in_channels=1, out_channels=16,
    ...                   kernel_size=3, padding=1)
    >>> x = lucid.zeros(2, 1, 16, 32, 32)   # (N, C, D, H, W)
    >>> y = conv3(x)
    >>> y.shape
    (2, 16, 16, 32, 32)

    Strided 3D convolution for spatial downsampling:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> conv3_stride = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
    >>> x = lucid.zeros(2, 16, 16, 32, 32)
    >>> y = conv3_stride(x)
    >>> y.shape
    (2, 32, 8, 16, 16)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d | str = 0,
        dilation: _Size3d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        kd, kh, kw = _triple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = _validate_padding_mode(padding_mode)
        if isinstance(padding, str):
            mode = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported(self.stride)
            self._padding_str: str | None = mode
            self.padding: tuple[int, int, int] = (0, 0, 0)
        else:
            self._padding_str = None
            self.padding = _triple(padding)
        self.weight = Parameter(
            empty(
                out_channels,
                in_channels // groups,
                kd,
                kh,
                kw,
                dtype=dtype,
                device=device,
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _resolve_pad(
        self, x: Tensor
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if self._padding_str == "valid":
            return (0, 0, 0), (0, 0, 0)
        if self._padding_str == "same":
            kd, kh, kw = self.kernel_size
            sd, sh, sw = self.stride
            dd, dh, dw = self.dilation
            lo_d, hi_d = _same_pad_pair(x.shape[2], kd, sd, dd)
            lo_h, hi_h = _same_pad_pair(x.shape[3], kh, sh, dh)
            lo_w, hi_w = _same_pad_pair(x.shape[4], kw, sw, dw)
            return (lo_d, lo_h, lo_w), (hi_d, hi_h, hi_w)
        return self.padding, self.padding

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        pad_lo, pad_hi = self._resolve_pad(x)
        return _conv_forward_with_mode(
            x,
            self.weight,
            self.bias,
            self.stride,
            pad_lo,
            pad_hi,
            self.dilation,
            self.groups,
            self.padding_mode,
            conv3d,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class ConvTranspose1d(Module):
    r"""Applies a 1D transposed convolution (fractionally-strided convolution).

    A transposed convolution — sometimes loosely called a *deconvolution*,
    though it is not a true mathematical inverse — is the gradient of a
    standard convolution with respect to its input.  It upsamples the
    spatial dimension by inserting implicit zeros between input elements
    before applying a convolution, and is the go-to building block for
    decoders, generators, and any network that must increase sequence
    length.

    Formally, for a stride :math:`s` and kernel size :math:`K`:

    .. math::

        L_{\text{out}} = (L_{\text{in}} - 1) \cdot s
                         - 2p + d(K - 1) + p_{\text{out}} + 1

    where :math:`p` is ``padding``, :math:`d` is ``dilation``, and
    :math:`p_{\text{out}}` is ``output_padding``.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input.
    out_channels : int
        Number of channels produced by the transposed convolution.
    kernel_size : int
        Size of the convolving kernel.
    stride : int, optional
        Stride of the convolution.  Values ``> 1`` upsample the input.
        Default: ``1``.
    padding : int, optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding is added
        to both sides of each dimension in the input.  Default: ``0``.
    output_padding : int, optional
        Additional size added to one side of the output shape.  Must
        satisfy ``0 <= output_padding < max(stride, dilation)``.  Used
        to disambiguate the output size when the formula
        :math:`(L_{\text{in}} - 1) \cdot s - 2p + d(K-1) + 1` is
        compatible with multiple :math:`L_{\text{in}}`. Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias. Default: ``True``.
    dilation : int, optional
        Spacing between kernel elements. Default: ``1``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable kernel of shape
        ``(in_channels, out_channels // groups, kernel_size)``.
        Note the channel axis ordering is **transposed** relative to
        :class:`Conv1d`: the leading dimension corresponds to
        ``in_channels``, not ``out_channels``.
        Initialized with Kaiming uniform (:math:`a = \sqrt{5}`).
    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, L)`
    Output:
        :math:`(N, C_{\text{out}}, L_{\text{out}})` where

        .. math::

            L_{\text{out}} = (L - 1) \cdot s - 2p + d(K - 1)
                             + p_{\text{out}} + 1

    Notes
    -----
    **Not a true inverse.**  :class:`ConvTranspose1d` is the transpose
    (adjoint) of :class:`Conv1d` in the linear-algebra sense: if
    ``conv_forward`` maps :math:`\mathbb{R}^{C_{\text{in}} \times L}
    \to \mathbb{R}^{C_{\text{out}} \times L'}`, then
    ``ConvTranspose1d`` with the same weights maps back
    :math:`\mathbb{R}^{C_{\text{out}} \times L'} \to
    \mathbb{R}^{C_{\text{in}} \times L}`.  When used in autoencoders or
    flow models the weights are learned independently and no exact
    inversion is implied.

    **output_padding.**  The transposed convolution output size formula
    can map multiple ``L_in`` values to the same ``L_out``.
    ``output_padding`` breaks this ambiguity by adding a single extra row
    on the output; it does *not* add actual padding to the input.

    Examples
    --------
    Upsample a sequence by factor 2:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> upsample = nn.ConvTranspose1d(in_channels=16, out_channels=16,
    ...                               kernel_size=4, stride=2, padding=1)
    >>> x = lucid.zeros(2, 16, 32)
    >>> y = upsample(x)
    >>> y.shape
    (2, 16, 64)

    Simple 1D decoder layer:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1)
    >>> x = lucid.zeros(4, 64, 20)
    >>> y = decoder(x)
    >>> y.shape
    (4, 32, 20)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        _validate_int_padding(padding, "ConvTranspose1d")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.weight = Parameter(
            empty(
                in_channels,
                out_channels // groups,
                kernel_size,
                dtype=dtype,
                device=device,
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return conv_transpose1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


class ConvTranspose2d(Module):
    r"""Applies a 2D transposed convolution (fractionally-strided convolution).

    Also known as a *fractionally-strided convolution*, this module is
    commonly used as the spatial upsampling primitive in generative
    models (VAEs, GANs), dense prediction decoders (U-Net), and
    super-resolution networks.  It is the transpose (adjoint) of
    :class:`Conv2d`.

    The output spatial dimensions satisfy:

    .. math::

        H_{\text{out}} = (H_{\text{in}} - 1) \cdot s_h - 2p_h
                         + d_h(K_H - 1) + p^{\text{out}}_h + 1

        W_{\text{out}} = (W_{\text{in}} - 1) \cdot s_w - 2p_w
                         + d_w(K_W - 1) + p^{\text{out}}_w + 1

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map.
    out_channels : int
        Number of channels produced by the transposed convolution.
    kernel_size : int or tuple[int, int]
        Size of the convolving kernel.
    stride : int or tuple[int, int], optional
        Stride.  Values ``> 1`` upsample the spatial dimensions.
        Default: ``1``.
    padding : int or tuple[int, int], optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding is added
        to both sides of each spatial dimension.  Default: ``0``.
    output_padding : int or tuple[int, int], optional
        Additional size added to one side of each spatial dimension of
        the output.  Must satisfy
        ``0 <= output_padding < max(stride, dilation)`` along each axis.
        Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias. Default: ``True``.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements. Default: ``1``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable kernel of shape
        ``(in_channels, out_channels // groups, K_H, K_W)``.
        The leading axis is ``in_channels`` — the reverse of
        :class:`Conv2d`.
        Initialized with Kaiming uniform (:math:`a = \sqrt{5}`).
    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, H, W)`
    Output:
        :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
        as given by the formulas above.

    Notes
    -----
    **Checkerboard artefacts.**  Transposed convolutions with
    ``stride > 1`` can produce characteristic checkerboard patterns in
    the output when kernel size is not divisible by stride.  A common
    mitigation is to use ``kernel_size = stride * n`` for some integer
    ``n``, or to replace the transposed conv with bilinear upsampling
    followed by a regular convolution.

    **output_padding.**  When ``stride > 1`` the output size formula is
    not injective: multiple input sizes map to the same output size.
    ``output_padding`` resolves this ambiguity and must be set
    consistently with the encoder stride to reconstruct the exact spatial
    dimensions.

    Examples
    --------
    VAE decoder: upsample 4×4 latent to 8×8:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder = nn.ConvTranspose2d(
    ...     in_channels=128, out_channels=64,
    ...     kernel_size=4, stride=2, padding=1
    ... )
    >>> z = lucid.zeros(8, 128, 4, 4)
    >>> y = decoder(z)
    >>> y.shape
    (8, 64, 8, 8)

    U-Net upsampling block:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    >>> x = lucid.zeros(4, 256, 16, 16)
    >>> y = up(x)
    >>> y.shape
    (4, 128, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d = 0,
        output_padding: _Size2d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size2d = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        _validate_int_padding(padding, "ConvTranspose2d")
        kh, kw = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.weight = Parameter(
            empty(
                in_channels, out_channels // groups, kh, kw, dtype=dtype, device=device
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


class ConvTranspose3d(Module):
    r"""Applies a 3D transposed convolution (fractionally-strided convolution).

    The 3D extension of :class:`ConvTranspose2d`.  It upsamples all
    three spatial dimensions simultaneously and is used in volumetric
    decoders such as 3D autoencoders, video generators, and medical
    image synthesis.

    Output size along each spatial axis :math:`X \in \{D, H, W\}`:

    .. math::

        X_{\text{out}} = (X_{\text{in}} - 1) \cdot s_x - 2p_x
                         + d_x(K_X - 1) + p^{\text{out}}_x + 1

    Parameters
    ----------
    in_channels : int
        Number of channels in the input volume.
    out_channels : int
        Number of channels produced by the transposed convolution.
    kernel_size : int or tuple[int, int, int]
        Size of the 3D convolving kernel.
    stride : int or tuple[int, int, int], optional
        Stride.  Values ``> 1`` upsample the spatial dimensions.
        Default: ``1``.
    padding : int or tuple[int, int, int], optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding applied
        on both sides of each axis.  Default: ``0``.
    output_padding : int or tuple[int, int, int], optional
        Additional size added to one side of each spatial dimension.
        Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias. Default: ``True``.
    dilation : int or tuple[int, int, int], optional
        Spacing between kernel elements. Default: ``1``.
    device : DeviceLike, optional
        Device on which to allocate parameters. Default: ``None``.
    dtype : DTypeLike, optional
        Data type for the parameters. Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable kernel of shape
        ``(in_channels, out_channels // groups, K_D, K_H, K_W)``.
        Leading axis is ``in_channels`` (same convention as
        :class:`ConvTranspose2d`).
        Initialized with Kaiming uniform (:math:`a = \sqrt{5}`).
    bias : Parameter or None
        Learnable bias of shape ``(out_channels,)``, or ``None``.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, D, H, W)`
    Output:
        :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
        as given by the formula above.

    Notes
    -----
    **Memory.**  3D transposed convolutions produce large feature maps
    at decoder stages.  Gradient checkpointing or smaller
    ``out_channels`` values are often necessary when operating on
    high-resolution volumes.

    **Symmetric decoder design.**  Pair each :class:`Conv3d` in the
    encoder with a :class:`ConvTranspose3d` having identical
    ``kernel_size``, ``stride``, and ``padding`` in the decoder to
    guarantee exact shape reconstruction.

    Examples
    --------
    Volumetric upsampling (2× along all spatial axes):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> up3d = nn.ConvTranspose3d(
    ...     in_channels=64, out_channels=32,
    ...     kernel_size=4, stride=2, padding=1
    ... )
    >>> x = lucid.zeros(2, 64, 4, 8, 8)
    >>> y = up3d(x)
    >>> y.shape
    (2, 32, 8, 16, 16)

    3D autoencoder decoder block:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder3d = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1, padding=1)
    >>> x = lucid.zeros(1, 128, 8, 8, 8)
    >>> y = decoder3d(x)
    >>> y.shape
    (1, 64, 8, 8, 8)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d = 0,
        output_padding: _Size3d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size3d = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        _validate_int_padding(padding, "ConvTranspose3d")
        kd, kh, kw = _triple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)
        self.groups = groups
        self.dilation = _triple(dilation)
        self.weight = Parameter(
            empty(
                in_channels,
                out_channels // groups,
                kd,
                kh,
                kw,
                dtype=dtype,
                device=device,
            )
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return conv_transpose3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


# ── Lazy convolutions ─────────────────────────────────────────────────────────
#
# Each lazy variant inherits from its eager counterpart so the forward path
# (`_resolve_pad` + `_conv_forward_with_mode`) is reused unchanged.  Parent
# `__init__` is intentionally skipped via `Module.__init__(self)` because it
# requires `in_channels`, which is what we are deferring.  The first forward
# call (or `_load_from_state_dict`) materialises the real `weight` / `bias`
# Parameter objects.


def _init_lazy_conv_weights(weight: Parameter, bias: Parameter | None) -> None:
    """Shared kaiming-uniform initialiser for lazily-built conv weights."""
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound: float = 1.0 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)


class LazyConv1d(Conv1d):
    r"""A :class:`Conv1d` that infers ``in_channels`` from the first input.

    Lazy modules defer the allocation and initialization of ``weight``
    and ``bias`` until the first call to :meth:`forward` (or until a
    compatible ``state_dict`` is loaded).  This removes the need to know
    ``in_channels`` at construction time, simplifying sequential model
    building and automatic architecture search.

    Materialization happens exactly once: on the first :meth:`forward`
    call the channel count is read from ``x.shape[1]``, weights are
    allocated and Kaiming-uniform initialized, and subsequent calls
    behave identically to a fully-initialized :class:`Conv1d`.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int
        Length of the 1D convolving kernel.
    stride : int, optional
        Stride of the convolution. Default: ``1``.
    padding : int or str, optional
        Zero-padding or ``"same"`` / ``"valid"`` string specifier.
        Default: ``0``.
    dilation : int, optional
        Spacing between kernel elements. Default: ``1``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device used when allocating weights at materialization time.
        Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` until first :meth:`forward`; afterwards shape
        ``(out_channels, in_channels // groups, kernel_size)``.
    bias : Parameter or None
        ``None`` until first :meth:`forward`; afterwards shape
        ``(out_channels,)`` if ``bias=True``, else remains ``None``.
    in_channels : int or None
        ``None`` before materialization; set to the inferred value on
        the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, L)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, L_{\text{out}})` — same formula as
        :class:`Conv1d`.

    Notes
    -----
    **State-dict loading.**  Loading a ``state_dict`` that contains a
    ``weight`` key of shape
    ``(out_channels, in_channels // groups, kernel_size)`` also triggers
    materialization, so a lazy module can be restored from a checkpoint
    without ever calling :meth:`forward`.

    **groups constraint.**  ``in_channels`` (inferred at runtime) must
    be divisible by ``groups``; this is checked inside ``_initialize``.

    Examples
    --------
    Lazy conv in a sequential model:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Sequential(
    ...     nn.LazyConv1d(out_channels=32, kernel_size=3, padding=1),
    ...     nn.ReLU(),
    ... )
    >>> x = lucid.zeros(4, 16, 64)   # in_channels=16 inferred here
    >>> y = model(x)
    >>> y.shape
    (4, 32, 64)

    Check that in_channels was inferred:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConv1d(out_channels=8, kernel_size=5, padding=2)
    >>> print(lazy.in_channels)
    None
    >>> _ = lazy(lucid.zeros(1, 3, 20))
    >>> print(lazy.in_channels)
    3
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        self.in_channels: int | None = None  # type: ignore[assignment]
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.dilation: int = dilation
        self.groups: int = groups
        self.padding_mode: str = _validate_padding_mode(padding_mode)
        self._padding_str: str | None
        if isinstance(padding, str):
            mode: str = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported((stride,))
            self._padding_str = mode
            self.padding: int = 0
        else:
            self._padding_str = None
            self.padding = padding
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        self.weight = Parameter(
            empty(
                self.out_channels,
                in_channels // self.groups,
                self.kernel_size,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 3:
                    error_msgs.append(
                        f"LazyConv1d expected 3-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[0]) != self.out_channels:
                    error_msgs.append(
                        f"LazyConv1d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels}, got {int(weight.shape[0])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[1]) * self.groups)
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return Conv1d.forward(self, x)

    def extra_repr(self) -> str:
        s: str = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class LazyConv2d(Conv2d):
    r"""A :class:`Conv2d` that infers ``in_channels`` from the first input.

    Lazy initialization allows building 2D convolutional networks without
    knowing the input channel count in advance.  The module registers
    ``weight`` and ``bias`` as ``None`` parameters and allocates them on
    the first :meth:`forward` call, reading ``C_in`` from
    ``x.shape[1]``.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int]
        Size of the 2D convolving kernel.
    stride : int or tuple[int, int], optional
        Stride of the convolution. Default: ``1``.
    padding : int, tuple[int, int], or str, optional
        Zero-padding or ``"same"`` / ``"valid"`` string specifier.
        Default: ``0``.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements. Default: ``1``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(out_channels, in_channels // groups, K_H, K_W)`` afterwards.
    bias : Parameter or None
        ``None`` before materialization; shape ``(out_channels,)`` if
        ``bias=True``, else remains ``None``.
    in_channels : int or None
        ``None`` before the first forward pass; inferred channel count
        afterwards.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, H, W)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` —
        same formula as :class:`Conv2d`.

    Notes
    -----
    **State-dict materialization.**  Loading a ``state_dict`` with a
    4-D ``weight`` tensor triggers materialization before parameter
    copying, enabling round-trip checkpoint compatibility.

    Examples
    --------
    Dynamic channel inference in a feature extractor:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Sequential(
    ...     nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1),
    ...     nn.ReLU(),
    ...     nn.LazyConv2d(out_channels=128, kernel_size=3, padding=1),
    ... )
    >>> x = lucid.zeros(2, 3, 32, 32)   # in_channels=3 inferred here
    >>> y = model(x)
    >>> y.shape
    (2, 128, 32, 32)

    Verify inferred in_channels after forward:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConv2d(out_channels=16, kernel_size=3, padding=1)
    >>> _ = lazy(lucid.zeros(1, 8, 16, 16))
    >>> print(lazy.in_channels)
    8
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d | str = 0,
        dilation: _Size2d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        self.in_channels: int | None = None  # type: ignore[assignment]
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = _pair(kernel_size)
        self.stride: tuple[int, int] = _pair(stride)
        self.dilation: tuple[int, int] = _pair(dilation)
        self.groups: int = groups
        self.padding_mode: str = _validate_padding_mode(padding_mode)
        self._padding_str: str | None
        if isinstance(padding, str):
            mode: str = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported(self.stride)
            self._padding_str = mode
            self.padding: tuple[int, int] = (0, 0)
        else:
            self._padding_str = None
            self.padding = _pair(padding)
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        kh, kw = self.kernel_size
        self.weight = Parameter(
            empty(
                self.out_channels,
                in_channels // self.groups,
                kh,
                kw,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 4:
                    error_msgs.append(
                        f"LazyConv2d expected 4-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[0]) != self.out_channels:
                    error_msgs.append(
                        f"LazyConv2d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels}, got {int(weight.shape[0])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[1]) * self.groups)
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return Conv2d.forward(self, x)

    def extra_repr(self) -> str:
        s: str = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class LazyConv3d(Conv3d):
    r"""A :class:`Conv3d` that infers ``in_channels`` from the first input.

    Lazy initialization for 3D convolution: ``in_channels`` need not be
    specified at construction time.  Weight allocation and Kaiming
    uniform initialization are deferred until the first :meth:`forward`
    call (or ``state_dict`` load), at which point ``C_in`` is read from
    ``x.shape[1]``.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int, int]
        Size of the 3D convolving kernel.
    stride : int or tuple[int, int, int], optional
        Stride of the convolution. Default: ``1``.
    padding : int, tuple[int, int, int], or str, optional
        Zero-padding or ``"same"`` / ``"valid"`` string specifier.
        Default: ``0``.
    dilation : int or tuple[int, int, int], optional
        Spacing between kernel elements. Default: ``1``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    padding_mode : str, optional
        ``"zeros"``, ``"reflect"``, ``"replicate"``, or ``"circular"``.
        Default: ``"zeros"``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(out_channels, in_channels // groups, K_D, K_H, K_W)`` afterwards.
    bias : Parameter or None
        ``None`` before materialization; ``(out_channels,)`` if
        ``bias=True``, else ``None``.
    in_channels : int or None
        ``None`` before the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, D, H, W)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})` —
        same formula as :class:`Conv3d`.

    Notes
    -----
    **State-dict materialization.**  A 5-D ``weight`` tensor in the
    ``state_dict`` triggers materialization before parameter copying.

    Examples
    --------
    Lazy 3D conv in a volumetric network:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy3d = nn.LazyConv3d(out_channels=32, kernel_size=3, padding=1)
    >>> x = lucid.zeros(2, 4, 16, 32, 32)   # in_channels=4 inferred
    >>> y = lazy3d(x)
    >>> y.shape
    (2, 32, 16, 32, 32)

    Verify materialization:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy3d = nn.LazyConv3d(out_channels=16, kernel_size=3, padding=1)
    >>> print(lazy3d.in_channels)
    None
    >>> _ = lazy3d(lucid.zeros(1, 8, 4, 4, 4))
    >>> print(lazy3d.in_channels)
    8
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d | str = 0,
        dilation: _Size3d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        self.in_channels: int | None = None  # type: ignore[assignment]
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int, int] = _triple(kernel_size)
        self.stride: tuple[int, int, int] = _triple(stride)
        self.dilation: tuple[int, int, int] = _triple(dilation)
        self.groups: int = groups
        self.padding_mode: str = _validate_padding_mode(padding_mode)
        self._padding_str: str | None
        if isinstance(padding, str):
            mode: str = padding.lower()
            if mode not in {"same", "valid"}:
                raise ValueError(
                    f"string padding must be 'same' or 'valid', got {padding!r}"
                )
            if mode == "same":
                _check_same_supported(self.stride)
            self._padding_str = mode
            self.padding: tuple[int, int, int] = (0, 0, 0)
        else:
            self._padding_str = None
            self.padding = _triple(padding)
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        kd, kh, kw = self.kernel_size
        self.weight = Parameter(
            empty(
                self.out_channels,
                in_channels // self.groups,
                kd,
                kh,
                kw,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 5:
                    error_msgs.append(
                        f"LazyConv3d expected 5-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[0]) != self.out_channels:
                    error_msgs.append(
                        f"LazyConv3d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels}, got {int(weight.shape[0])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[1]) * self.groups)
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return Conv3d.forward(self, x)

    def extra_repr(self) -> str:
        s: str = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self._padding_str if self._padding_str else self.padding}"
        )
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


# ── Lazy ConvTranspose ────────────────────────────────────────────────────────
# Weight layout is (in_channels, out_channels // groups, *K).  The lazy
# dimension is ``in_channels`` — the leading axis of the saved weight — so
# materialisation reads ``weight.shape[0]``.


class LazyConvTranspose1d(ConvTranspose1d):
    r"""A :class:`ConvTranspose1d` that infers ``in_channels`` from the first input.

    Combines the fractionally-strided upsampling of
    :class:`ConvTranspose1d` with lazy weight materialization.
    ``in_channels`` need not be known at construction time; it is read
    from ``x.shape[1]`` on the first :meth:`forward` call.

    Parameters
    ----------
    out_channels : int
        Number of output channels after the transposed convolution.
    kernel_size : int
        Length of the 1D convolving kernel.
    stride : int, optional
        Stride (upsampling factor). Default: ``1``.
    padding : int, optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding on each
        side.  String padding is not supported. Default: ``0``.
    output_padding : int, optional
        Additional size added to one side of the output. Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    dilation : int, optional
        Spacing between kernel elements. Default: ``1``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(in_channels, out_channels // groups, kernel_size)`` afterwards.
        Note: leading axis is ``in_channels``, not ``out_channels``.
    bias : Parameter or None
        ``None`` before materialization; ``(out_channels,)`` if
        ``bias=True``.
    in_channels : int or None
        ``None`` before the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, L)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, L_{\text{out}})` — same formula as
        :class:`ConvTranspose1d`.

    Notes
    -----
    **State-dict loading.**  A 3-D ``weight`` tensor in the
    ``state_dict`` whose leading axis equals the inferred ``in_channels``
    triggers materialization before parameter copying.

    Examples
    --------
    Lazy transposed conv in a sequence decoder:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder = nn.LazyConvTranspose1d(
    ...     out_channels=16, kernel_size=4, stride=2, padding=1
    ... )
    >>> x = lucid.zeros(2, 32, 10)   # in_channels=32 inferred
    >>> y = decoder(x)
    >>> y.shape
    (2, 16, 20)

    Inspect pre- and post-materialization state:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConvTranspose1d(out_channels=8, kernel_size=3, padding=1)
    >>> print(lazy.in_channels)
    None
    >>> _ = lazy(lucid.zeros(1, 4, 16))
    >>> print(lazy.in_channels)
    4
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        _validate_int_padding(padding, "LazyConvTranspose1d")
        self.in_channels: int | None = None  # type: ignore[assignment]
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding
        self.output_padding: int = output_padding
        self.groups: int = groups
        self.dilation: int = dilation
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        self.weight = Parameter(
            empty(
                in_channels,
                self.out_channels // self.groups,
                self.kernel_size,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 3:
                    error_msgs.append(
                        f"LazyConvTranspose1d expected 3-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[1]) != self.out_channels // self.groups:
                    error_msgs.append(
                        f"LazyConvTranspose1d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels // self.groups}, "
                        f"got {int(weight.shape[1])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[0]))
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return ConvTranspose1d.forward(self, x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


class LazyConvTranspose2d(ConvTranspose2d):
    r"""A :class:`ConvTranspose2d` that infers ``in_channels`` from the first input.

    Lazy version of the 2D transposed convolution.  Weight allocation is
    deferred until the first :meth:`forward` call, making it easy to
    compose upsampling decoder networks without manually tracking the
    channel dimension through each block.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int]
        Size of the 2D convolving kernel.
    stride : int or tuple[int, int], optional
        Stride (upsampling factor). Default: ``1``.
    padding : int or tuple[int, int], optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding applied
        on each side.  String padding is not supported. Default: ``0``.
    output_padding : int or tuple[int, int], optional
        Additional size added to one side of each output dimension.
        Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements. Default: ``1``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(in_channels, out_channels // groups, K_H, K_W)`` afterwards.
    bias : Parameter or None
        ``None`` before materialization; ``(out_channels,)`` if
        ``bias=True``.
    in_channels : int or None
        ``None`` before the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, H, W)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` —
        same formula as :class:`ConvTranspose2d`.

    Notes
    -----
    **State-dict loading.**  A 4-D ``weight`` tensor (shape
    ``(in_channels, out_channels // groups, K_H, K_W)``) triggers
    materialization before parameter copying.

    **Symmetric encoder–decoder pairs.**  Pair each :class:`LazyConv2d`
    encoder stride with a :class:`LazyConvTranspose2d` of matching
    ``kernel_size``, ``stride``, and ``padding`` to reconstruct exact
    spatial dimensions.

    Examples
    --------
    Lazy VAE decoder:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> decoder = nn.LazyConvTranspose2d(
    ...     out_channels=64, kernel_size=4, stride=2, padding=1
    ... )
    >>> z = lucid.zeros(4, 128, 8, 8)   # in_channels=128 inferred
    >>> y = decoder(z)
    >>> y.shape
    (4, 64, 16, 16)

    Verify inference:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConvTranspose2d(out_channels=32, kernel_size=3)
    >>> _ = lazy(lucid.zeros(1, 64, 4, 4))
    >>> print(lazy.in_channels)
    64
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size2d,
        stride: _Size2d = 1,
        padding: _Size2d = 0,
        output_padding: _Size2d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size2d = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        _validate_int_padding(padding, "LazyConvTranspose2d")
        self.in_channels: int | None = None  # type: ignore[assignment]
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = _pair(kernel_size)
        self.stride: tuple[int, int] = _pair(stride)
        self.padding: tuple[int, int] = _pair(padding)
        self.output_padding: tuple[int, int] = _pair(output_padding)
        self.groups: int = groups
        self.dilation: tuple[int, int] = _pair(dilation)
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        kh, kw = self.kernel_size
        self.weight = Parameter(
            empty(
                in_channels,
                self.out_channels // self.groups,
                kh,
                kw,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 4:
                    error_msgs.append(
                        f"LazyConvTranspose2d expected 4-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[1]) != self.out_channels // self.groups:
                    error_msgs.append(
                        f"LazyConvTranspose2d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels // self.groups}, "
                        f"got {int(weight.shape[1])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[0]))
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return ConvTranspose2d.forward(self, x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


class LazyConvTranspose3d(ConvTranspose3d):
    r"""A :class:`ConvTranspose3d` that infers ``in_channels`` from the first input.

    Lazy version of the 3D transposed convolution.  Combines volumetric
    upsampling with deferred weight allocation, simplifying the
    construction of 3D generative decoders and medical image synthesis
    networks where channel dimensions are computed dynamically.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int, int]
        Size of the 3D convolving kernel.
    stride : int or tuple[int, int, int], optional
        Stride (upsampling factor). Default: ``1``.
    padding : int or tuple[int, int, int], optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding applied
        on each side of each spatial axis.  String padding is not
        supported. Default: ``0``.
    output_padding : int or tuple[int, int, int], optional
        Additional size added to one side of each spatial output
        dimension. Default: ``0``.
    groups : int, optional
        Number of blocked connections. Default: ``1``.
    bias : bool, optional
        If ``True``, a learnable bias is added after materialization.
        Default: ``True``.
    dilation : int or tuple[int, int, int], optional
        Spacing between kernel elements. Default: ``1``.
    device : DeviceLike, optional
        Device used when allocating weights. Default: ``None``.
    dtype : DTypeLike, optional
        Data type used when allocating weights. Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before materialization; shape
        ``(in_channels, out_channels // groups, K_D, K_H, K_W)``
        afterwards.
    bias : Parameter or None
        ``None`` before materialization; ``(out_channels,)`` if
        ``bias=True``.
    in_channels : int or None
        ``None`` before the first forward pass.

    Shape
    -----
    Input:
        :math:`(N, C_{\text{in}}, D, H, W)` — :math:`C_{\text{in}}` is
        inferred automatically.
    Output:
        :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})` —
        same formula as :class:`ConvTranspose3d`.

    Notes
    -----
    **State-dict loading.**  A 5-D ``weight`` tensor triggers
    materialization before parameter copying.

    **Symmetric 3D encoder–decoder.**  Pair each :class:`LazyConv3d`
    with a :class:`LazyConvTranspose3d` using matching ``kernel_size``,
    ``stride``, and ``padding`` for lossless spatial reconstruction.

    Examples
    --------
    Lazy 3D decoder block:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> up3d = nn.LazyConvTranspose3d(
    ...     out_channels=32, kernel_size=4, stride=2, padding=1
    ... )
    >>> x = lucid.zeros(2, 64, 4, 8, 8)   # in_channels=64 inferred
    >>> y = up3d(x)
    >>> y.shape
    (2, 32, 8, 16, 16)

    Verify materialization:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lazy = nn.LazyConvTranspose3d(out_channels=16, kernel_size=3, padding=1)
    >>> print(lazy.in_channels)
    None
    >>> _ = lazy(lucid.zeros(1, 8, 4, 4, 4))
    >>> print(lazy.in_channels)
    8
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: _Size3d,
        stride: _Size3d = 1,
        padding: _Size3d = 0,
        output_padding: _Size3d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _Size3d = 1,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        Module.__init__(self)
        _validate_int_padding(padding, "LazyConvTranspose3d")
        self.in_channels: int | None = None  # type: ignore[assignment]
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int, int] = _triple(kernel_size)
        self.stride: tuple[int, int, int] = _triple(stride)
        self.padding: tuple[int, int, int] = _triple(padding)
        self.output_padding: tuple[int, int, int] = _triple(output_padding)
        self.groups: int = groups
        self.dilation: tuple[int, int, int] = _triple(dilation)
        self._has_bias: bool = bias
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def _initialize(self, in_channels: int) -> None:
        self.in_channels = in_channels
        kd, kh, kw = self.kernel_size
        self.weight = Parameter(
            empty(
                in_channels,
                self.out_channels // self.groups,
                kd,
                kh,
                kw,
                dtype=self._dtype,
                device=self._device,
            )
        )
        if self._has_bias:
            self.bias = Parameter(
                empty(self.out_channels, dtype=self._dtype, device=self._device)
            )
        else:
            self.bias = None
        _init_lazy_conv_weights(self.weight, self.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.weight is None:
            weight: Tensor | None = state_dict.get(f"{prefix}weight")
            if weight is not None:
                if len(weight.shape) != 5:
                    error_msgs.append(
                        f"LazyConvTranspose3d expected 5-D weight at '{prefix}weight', "
                        f"got {tuple(weight.shape)}"
                    )
                    return
                if int(weight.shape[1]) != self.out_channels // self.groups:
                    error_msgs.append(
                        f"LazyConvTranspose3d out_channels mismatch at '{prefix}weight': "
                        f"expected {self.out_channels // self.groups}, "
                        f"got {int(weight.shape[1])}"
                    )
                    return
                self._dtype = self._dtype or weight.dtype
                self._device = self._device or weight.device
                self._initialize(int(weight.shape[0]))
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        if self.weight is None:
            self._initialize(int(x.shape[1]))
        return ConvTranspose3d.forward(self, x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )
