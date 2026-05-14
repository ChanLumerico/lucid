"""
nn.functional convolution operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _normalize_int_or_tuple(v: int | tuple[int, ...], n: int) -> tuple[int, ...]:
    if isinstance(v, int):
        return (v,) * n
    return tuple(v)


def conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    r"""1-D cross-correlation over batched 3-D input.

    Slides a learned filter along a single spatial axis.  As with every
    major deep-learning framework, the operation is technically
    cross-correlation rather than strict mathematical convolution (no
    kernel flip), but the "conv" name is kept for familiarity.  Channel
    mixing happens through the ``in_channels`` dimension of ``weight``.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C_in, L)``.
    weight : Tensor
        Filters of shape ``(C_out, C_in/groups, kL)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.
    stride : int or tuple of int, optional
        Step between adjacent kernel positions (default ``1``).
    padding : int or tuple of int, optional
        Zero padding applied to both sides of the spatial axis.
    dilation : int or tuple of int, optional
        Spacing between kernel taps (atrous convolution).  Default ``1``.
    groups : int, optional
        Split the channel dimension into ``groups`` independent groups.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, L_out)`` where

        .. math::

            L_{\text{out}} = \left\lfloor \frac{L + 2p - d(k - 1) - 1}{s} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,\,c_o,\,l} = b_{c_o} + \sum_{c_i, m} w_{c_o,\,c_i,\,m} \cdot x_{i,\,c_i,\,s l + d m}

    Backward is registered automatically; gradients flow to ``x``,
    ``weight``, and ``bias``.  Set ``groups == C_in`` and ``C_out == C_in``
    for a depthwise convolution.  Using ``dilation > 1`` enlarges the
    receptive field without increasing parameter count — common in
    speech / audio models (WaveNet) and 1-D temporal CNNs.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv1d
    >>> x = lucid.randn(2, 4, 50)
    >>> w = lucid.randn(8, 4, 3)
    >>> y = conv1d(x, w, padding=1)
    >>> y.shape
    (2, 8, 50)
    """
    s = _normalize_int_or_tuple(stride, 1)[0]
    p = _normalize_int_or_tuple(padding, 1)[0]
    d = _normalize_int_or_tuple(dilation, 1)[0]
    b = _unwrap(bias) if bias is not None else None
    return _wrap(_C_engine.nn.conv1d(_unwrap(x), _unwrap(weight), b, s, p, d, groups))


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    r"""2-D cross-correlation over batched 4-D input.

    Despite the name, computes **cross-correlation** rather than strict
    mathematical convolution (no kernel flip).  The kernel slides over
    the input applying a learned linear combination at each spatial
    position; channel mixing happens through the ``in_channels``
    dimension of ``weight``.  This is the workhorse op of modern image
    CNNs (ResNet, ConvNeXt, EfficientNet, ...).

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C_in, H, W)``.
    weight : Tensor
        Filters of shape ``(C_out, C_in/groups, kH, kW)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.
    stride : int or (int, int), optional
        Step between adjacent kernel positions (default ``1``).
    padding : int or (int, int), optional
        Zero padding on each spatial side.
    dilation : int or (int, int), optional
        Spacing between kernel taps (atrous convolution).  Default ``1``.
    groups : int, optional
        Split the channels into ``groups`` independent groups.  Setting
        ``groups == C_in`` gives a depthwise convolution.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, H_out, W_out)`` where

        .. math::

            H_{\text{out}} &= \left\lfloor \frac{H + 2 p_H - d_H (k_H - 1) - 1}{s_H} + 1 \right\rfloor \\
            W_{\text{out}} &= \left\lfloor \frac{W + 2 p_W - d_W (k_W - 1) - 1}{s_W} + 1 \right\rfloor

    Notes
    -----
    Math:

    .. math::

        y_{i,\,c_o,\,h,\,w} = b_{c_o} + \sum_{c_i, m, n} w_{c_o,\,c_i,\,m,\,n} \cdot x_{i,\,c_i,\,s_H h + d_H m,\,s_W w + d_W n}

    Backward is well-known; gradients w.r.t. ``x``, ``weight``, and
    ``bias`` flow through automatically.  ``groups > 1`` yields grouped
    convolution (channel-blocks computed independently); ``groups ==
    C_in`` plus ``C_out == C_in`` is depthwise convolution.  Dilation
    enlarges the receptive field without inflating parameter count.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv2d
    >>> x = lucid.randn(1, 3, 32, 32)
    >>> w = lucid.randn(8, 3, 3, 3)
    >>> y = conv2d(x, w, stride=1, padding=1)
    >>> y.shape
    (1, 8, 32, 32)
    """
    sh, sw = _normalize_int_or_tuple(stride, 2)
    ph, pw = _normalize_int_or_tuple(padding, 2)
    dh, dw = _normalize_int_or_tuple(dilation, 2)
    b = _unwrap(bias) if bias is not None else None
    return _wrap(
        _C_engine.nn.conv2d(
            _unwrap(x), _unwrap(weight), b, sh, sw, ph, pw, dh, dw, groups
        )
    )


def conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
) -> Tensor:
    r"""3-D cross-correlation over batched 5-D input.

    Extends :func:`conv2d` to volumetric data (depth × height × width).
    Standard in medical imaging (CT, MRI), video understanding (I3D,
    3D-ResNet, SlowFast) and any setting where the input has three
    spatial axes.  As with the 1-D and 2-D variants, this is technically
    cross-correlation rather than strict convolution.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C_in, D, H, W)``.
    weight : Tensor
        Filters of shape ``(C_out, C_in/groups, kD, kH, kW)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.
    stride : int or (int, int, int), optional
        Step between adjacent kernel positions per axis (default ``1``).
    padding : int or (int, int, int), optional
        Zero padding on each spatial side.
    dilation : int or (int, int, int), optional
        Spacing between kernel taps (atrous convolution).  Default ``1``.
    groups : int, optional
        Split channels into ``groups`` independent groups.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, D_out, H_out, W_out)`` where each
        spatial size obeys

        .. math::

            D_{\text{out}} = \left\lfloor \frac{D + 2 p_D - d_D (k_D - 1) - 1}{s_D} + 1 \right\rfloor

        (analogously for ``H`` and ``W``).

    Notes
    -----
    Math:

    .. math::

        y_{i,\,c_o,\,d,\,h,\,w} = b_{c_o} + \sum_{c_i,\,m,\,n,\,p} w_{c_o,\,c_i,\,m,\,n,\,p} \cdot x_{i,\,c_i,\,s_D d + d_D m,\,s_H h + d_H n,\,s_W w + d_W p}

    3-D convolution has cubic kernel cost in ``k`` — for large kernels,
    consider factorised variants (e.g. ``(1, k, k)`` followed by
    ``(k, 1, 1)``) which trade expressivity for compute.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv3d
    >>> x = lucid.randn(1, 1, 16, 32, 32)
    >>> w = lucid.randn(4, 1, 3, 3, 3)
    >>> y = conv3d(x, w, padding=1)
    >>> y.shape
    (1, 4, 16, 32, 32)
    """
    sd, sh, sw = _normalize_int_or_tuple(stride, 3)
    pd, ph, pw = _normalize_int_or_tuple(padding, 3)
    dd, dh, dw = _normalize_int_or_tuple(dilation, 3)
    b = _unwrap(bias) if bias is not None else None
    return _wrap(
        _C_engine.nn.conv3d(
            _unwrap(x), _unwrap(weight), b, sd, sh, sw, pd, ph, pw, dd, dh, dw, groups
        )
    )


def conv_transpose1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
    dilation: int | tuple[int, ...] = 1,
) -> Tensor:
    r"""Transposed 1-D convolution (a.k.a. "fractionally-strided" conv).

    Often called "deconvolution" in the literature, but this is **not**
    the mathematical inverse of :func:`conv1d` — it is the gradient
    operator of a forward convolution, used to map a low-resolution
    feature map back to a higher-resolution one.  The standard
    upsampling primitive in 1-D decoder networks (autoencoders, TTS
    vocoders, etc.).

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C_in, L_in)``.
    weight : Tensor
        Filters of shape ``(C_in, C_out/groups, kL)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.  When ``None``,
        a zero bias is materialised internally (the engine op requires
        an explicit bias tensor).
    stride : int or tuple of int, optional
        Upsampling factor between input cells (default ``1``).
    padding : int or tuple of int, optional
        Symmetric zero-padding shaved from each side of the output.
    output_padding : int or tuple of int, optional
        Extra one-sided trailing padding added to the output.  Used to
        disambiguate the output size when ``stride > 1``.
    groups : int, optional
        Channel grouping (currently ``1``-only in the engine path).
    dilation : int or tuple of int, optional
        Spacing between kernel taps.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, L_out)`` where

        .. math::

            L_{\text{out}} = (L_{\text{in}} - 1) \cdot s - 2 p + d (k - 1) + \text{op} + 1

    Notes
    -----
    Conceptually, transposed convolution inserts ``stride - 1`` zeros
    between consecutive input samples and then runs a normal
    convolution — hence the "fractional stride" name.  In practice the
    engine implements it as the matrix-transpose of the corresponding
    forward conv, which avoids materialising the zero-stuffed buffer.

    Checkerboard artifacts (regular high-frequency patterns) commonly
    arise when ``kernel_size`` is not divisible by ``stride``; an
    alternative is :func:`conv1d` after upsampling.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv_transpose1d
    >>> x = lucid.randn(1, 4, 10)
    >>> w = lucid.randn(4, 2, 3)     # (C_in, C_out, kL)
    >>> y = conv_transpose1d(x, w, stride=2)
    >>> y.shape
    (1, 2, 21)
    """
    s = _normalize_int_or_tuple(stride, 1)[0]
    p = _normalize_int_or_tuple(padding, 1)[0]
    op = _normalize_int_or_tuple(output_padding, 1)[0]
    wi = _unwrap(weight)
    # engine requires explicit bias; create zeros(C_out) when caller passes None
    if bias is not None:
        b = _unwrap(bias)
    else:
        c_out = int(wi.shape[1])
        b = _C_engine.zeros([c_out], wi.dtype, wi.device)
    return _wrap(_C_engine.nn.conv_transpose1d(_unwrap(x), wi, b, s, p, op))


def conv_transpose2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    groups: int = 1,
    dilation: int | tuple[int, int] = 1,
) -> Tensor:
    r"""Transposed 2-D convolution — the standard upsampling primitive.

    Also referred to as "deconvolution", though this is not the
    mathematical inverse of :func:`conv2d`.  Performs the transpose of
    a 2-D forward convolution: each input cell scatters a weighted
    copy of the kernel into the (larger) output canvas.  Used in
    image-segmentation decoders, GAN generators, super-resolution
    networks, and U-Net style architectures.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C_in, H_in, W_in)``.
    weight : Tensor
        Filters of shape ``(C_in, C_out/groups, kH, kW)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.  Materialised as
        zeros when ``None``.
    stride : int or (int, int), optional
        Upsampling factor between input cells (default ``1``).
    padding : int or (int, int), optional
        Implicit zero-padding subtracted from each side of the output.
    output_padding : int or (int, int), optional
        Additional one-sided trailing padding on the output, used to
        resolve the size ambiguity when ``stride > 1``.
    groups : int, optional
        Channel grouping.
    dilation : int or (int, int), optional
        Spacing between kernel taps.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, H_out, W_out)`` where each spatial
        dimension satisfies

        .. math::

            H_{\text{out}} = (H_{\text{in}} - 1) \cdot s_H - 2 p_H + d_H (k_H - 1) + \text{op}_H + 1

    Notes
    -----
    Checkerboard artifacts are a well-known failure mode when
    ``kernel_size`` is not a multiple of ``stride``.  Alternatives are
    nearest/bilinear upsample + :func:`conv2d`, or pixel-shuffle
    (sub-pixel convolution).

    The forward of this op equals the **backward of conv2d w.r.t.
    input**; that duality is what makes it useful as a learnable
    upsampler — its gradients give the structure of an ordinary conv.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv_transpose2d
    >>> x = lucid.randn(1, 16, 7, 7)
    >>> w = lucid.randn(16, 8, 4, 4)
    >>> y = conv_transpose2d(x, w, stride=2, padding=1)
    >>> y.shape
    (1, 8, 14, 14)
    """
    sh, sw = _normalize_int_or_tuple(stride, 2)
    ph, pw = _normalize_int_or_tuple(padding, 2)
    oh, ow = _normalize_int_or_tuple(output_padding, 2)
    wi = _unwrap(weight)
    if bias is not None:
        b = _unwrap(bias)
    else:
        b = _C_engine.zeros([int(wi.shape[1])], wi.dtype, wi.device)
    return _wrap(
        _C_engine.nn.conv_transpose2d(_unwrap(x), wi, b, sh, sw, ph, pw, oh, ow)
    )


def conv_transpose3d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    output_padding: int | tuple[int, int, int] = 0,
    groups: int = 1,
    dilation: int | tuple[int, int, int] = 1,
) -> Tensor:
    r"""Transposed 3-D convolution — volumetric upsampling.

    Extends :func:`conv_transpose2d` to three spatial dimensions.
    Standard in 3-D segmentation decoders (V-Net, 3D-UNet), volumetric
    GANs, and video generation.  Like the 1-D / 2-D variants, this is
    the matrix-transpose of an ordinary 3-D convolution rather than a
    true mathematical inverse.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C_in, D_in, H_in, W_in)``.
    weight : Tensor
        Filters of shape ``(C_in, C_out/groups, kD, kH, kW)``.
    bias : Tensor, optional
        Per-output-channel bias of shape ``(C_out,)``.  Materialised as
        zeros when ``None``.
    stride : int or (int, int, int), optional
        Upsampling factor per spatial axis.
    padding : int or (int, int, int), optional
        Implicit zero-padding subtracted from each side of the output.
    output_padding : int or (int, int, int), optional
        Trailing one-sided output padding.
    groups : int, optional
        Channel grouping.
    dilation : int or (int, int, int), optional
        Spacing between kernel taps.

    Returns
    -------
    Tensor
        Output of shape ``(N, C_out, D_out, H_out, W_out)`` where each
        spatial size obeys

        .. math::

            D_{\text{out}} = (D_{\text{in}} - 1) \cdot s_D - 2 p_D + d_D (k_D - 1) + \text{op}_D + 1

        (analogously for ``H`` and ``W``).

    Notes
    -----
    Memory cost scales cubically in the spatial dimensions; for large
    volumes a typical pattern is to interleave trilinear upsample with
    :func:`conv3d` instead.  Checkerboard artifacts also extend into
    three dimensions when ``kernel_size`` is not a multiple of
    ``stride``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import conv_transpose3d
    >>> x = lucid.randn(1, 8, 4, 8, 8)
    >>> w = lucid.randn(8, 4, 2, 2, 2)
    >>> y = conv_transpose3d(x, w, stride=2)
    >>> y.shape
    (1, 4, 8, 16, 16)
    """
    sd, sh, sw = _normalize_int_or_tuple(stride, 3)
    pd, ph, pw = _normalize_int_or_tuple(padding, 3)
    od, oh, ow = _normalize_int_or_tuple(output_padding, 3)
    wi = _unwrap(weight)
    if bias is not None:
        b = _unwrap(bias)
    else:
        b = _C_engine.zeros([int(wi.shape[1])], wi.dtype, wi.device)
    return _wrap(
        _C_engine.nn.conv_transpose3d(
            _unwrap(x), wi, b, sd, sh, sw, pd, ph, pw, od, oh, ow
        )
    )
