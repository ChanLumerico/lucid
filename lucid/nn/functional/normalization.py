"""
nn.functional normalization operations.
"""

from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def batch_norm(
    x: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Batch normalization (Ioffe & Szegedy, 2015).

    Normalises each channel using statistics computed across the batch
    and all spatial axes, then applies a learnable per-channel affine
    transform.  Acts as a strong regulariser and an enabler of higher
    learning rates by reducing "internal covariate shift".

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, *)`` with 2–5 dimensions:
        ``(N, C)``, ``(N, C, L)``, ``(N, C, H, W)``, or
        ``(N, C, D, H, W)``.
    running_mean : Tensor or None
        Running mean buffer of shape ``(C,)``; consulted in eval mode.
    running_var : Tensor or None
        Running variance buffer of shape ``(C,)``; consulted in eval mode.
    weight : Tensor, optional
        Per-channel scale :math:`\gamma` of shape ``(C,)``.  Defaults to
        ones (no scaling).
    bias : Tensor, optional
        Per-channel shift :math:`\beta` of shape ``(C,)``.  Defaults to
        zeros (no shift).
    training : bool, optional
        When ``True``, statistics come from the current batch; when
        ``False`` and running buffers are supplied, those are used.
    momentum : float, optional
        Exponential-moving-average coefficient for the running buffers.
    eps : float, optional
        Numerical safety added inside the square root.

    Returns
    -------
    Tensor
        Same shape as ``x``.

    Notes
    -----
    Math (over batch :math:`\mathcal{B}` and spatial dims):

    .. math::

        \mu_{\mathcal{B}} &= \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} x_i \\
        \sigma_{\mathcal{B}}^2 &= \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} (x_i - \mu_{\mathcal{B}})^2 \\
        \hat{x}_i &= \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \\
        y_i &= \gamma\,\hat{x}_i + \beta

    Dispatches internally by ``ndim`` to the matching engine kernel:
    ``ndim==2`` is treated as ``(N, C, 1)``; ``ndim==3`` uses
    ``batch_norm1d``; ``ndim==4`` uses the 2-D op; ``ndim==5`` uses
    ``batch_norm3d``.  Eval mode bypasses statistics computation
    entirely and uses ``running_mean`` / ``running_var`` directly.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import batch_norm
    >>> x = lucid.randn(8, 16, 32, 32)
    >>> rm = lucid.zeros(16); rv = lucid.ones(16)
    >>> y = batch_norm(x, rm, rv, training=True)
    >>> y.shape
    (8, 16, 32, 32)
    """
    from lucid._factories.creation import ones, zeros

    C = x.shape[1]
    ndim = x.ndim
    xi = _unwrap(x)
    w = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(C, device=x.device, dtype=x.dtype))
    )
    b = (
        _unwrap(bias)
        if bias is not None
        else _unwrap(zeros(C, device=x.device, dtype=x.dtype))
    )

    # ── Eval mode: use precomputed running statistics ─────────────────────────
    if not training and running_mean is not None and running_var is not None:
        rm = _unwrap(running_mean)
        rv = _unwrap(running_var)
        out_impl = _C_engine.nn.batch_norm_eval(xi, rm, rv, w, b, eps)
        return _wrap(out_impl)

    # ── Training mode: dispatch by dimensionality ─────────────────────────────
    if ndim == 2:
        # (N, C) → unsqueeze to (N, C, 1), batch_norm1d, squeeze back
        xi_3d = _C_engine.unsqueeze(xi, 2)
        out_3d = _C_engine.nn.batch_norm1d(xi_3d, w, b, eps)
        return _wrap(_C_engine.squeeze(out_3d, 2))
    elif ndim == 3:
        return _wrap(_C_engine.nn.batch_norm1d(xi, w, b, eps))
    elif ndim == 4:
        return _wrap(_C_engine.nn.batch_norm(xi, w, b, eps))  # type: ignore[call-arg, arg-type]
    elif ndim == 5:
        return _wrap(_C_engine.nn.batch_norm3d(xi, w, b, eps))
    else:
        raise ValueError(f"batch_norm: expected 2–5D input, got ndim={ndim}")


def layer_norm(
    x: Tensor,
    normalized_shape: list[int] | tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Layer normalization (Ba, Kiros & Hinton, 2016).

    Normalises each sample independently across the **last**
    ``normalized_shape`` dimensions.  Unlike :func:`batch_norm`, no
    batch statistics are involved — making LayerNorm the default
    choice for transformers and other models where batches may be
    small or sequence lengths variable.

    Parameters
    ----------
    x : Tensor
        Input whose trailing dims match ``normalized_shape``.
    normalized_shape : list of int or tuple of int
        Trailing dims to normalise over.  E.g. ``(d,)`` for a
        token-wise normalisation in a transformer with hidden size
        ``d``.
    weight : Tensor, optional
        Per-element scale :math:`\gamma` of shape ``normalized_shape``.
        Defaults to ones.
    bias : Tensor, optional
        Per-element shift :math:`\beta` of shape ``normalized_shape``.
        Defaults to zeros.
    eps : float, optional
        Numerical safety added inside the square root.

    Returns
    -------
    Tensor
        Same shape as ``x``.

    Notes
    -----
    Math (reduction taken over the last :math:`k` axes,
    :math:`k = \mathrm{len}(\text{normalized\_shape})`):

    .. math::

        \mu &= \frac{1}{|S|}\sum_{j \in S} x_j \\
        \sigma^2 &= \frac{1}{|S|}\sum_{j \in S} (x_j - \mu)^2 \\
        y &= \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

    Because the reduction is per-sample, behaviour is identical at
    train and eval time — no running statistics needed.  This is what
    makes LayerNorm so prevalent in sequence models (RNNs,
    transformers).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import layer_norm
    >>> x = lucid.randn(2, 10, 64)
    >>> y = layer_norm(x, normalized_shape=(64,))
    >>> y.shape
    (2, 10, 64)
    """
    from lucid._factories.creation import ones, zeros

    shape: tuple[int, ...] = tuple(normalized_shape)
    w: _C_engine.TensorImpl = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(*shape, device=x.device, dtype=x.dtype))
    )
    b: _C_engine.TensorImpl = (
        _unwrap(bias)
        if bias is not None
        else _unwrap(zeros(*shape, device=x.device, dtype=x.dtype))
    )
    # Engine API: layer_norm(x, gamma, beta, eps) — no normalized_shape arg
    return _wrap(_C_engine.nn.layer_norm(_unwrap(x), w, b, eps))


def group_norm(
    x: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Group normalization (Wu & He, 2018).

    Splits the channel dimension into ``num_groups`` contiguous groups
    and normalises each ``(sample, group)`` slice independently across
    its channels and spatial axes.  Combines the spatial reduction of
    BatchNorm with the per-sample stability of LayerNorm — performance
    is therefore largely independent of batch size, which makes it the
    go-to choice for detection / segmentation models trained with very
    small batches.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, *spatial)`` where ``C`` must be
        divisible by ``num_groups``.
    num_groups : int
        Number of channel groups.  Two limiting cases: ``num_groups ==
        C`` reduces to InstanceNorm; ``num_groups == 1`` reduces to
        LayerNorm over channels + spatial axes.
    weight : Tensor, optional
        Per-channel scale :math:`\gamma` of shape ``(C,)``.
    bias : Tensor, optional
        Per-channel shift :math:`\beta` of shape ``(C,)``.
    eps : float, optional
        Numerical safety added inside the square root.

    Returns
    -------
    Tensor
        Same shape as ``x``.

    Notes
    -----
    Math (let :math:`G_g` be the channel set of group :math:`g` and
    :math:`S` the spatial axes):

    .. math::

        \mu_{n,g} &= \frac{1}{|G_g||S|} \sum_{c \in G_g} \sum_{s \in S} x_{n,c,s} \\
        \sigma^2_{n,g} &= \frac{1}{|G_g||S|} \sum_{c \in G_g} \sum_{s \in S} (x_{n,c,s} - \mu_{n,g})^2 \\
        y_{n,c,s} &= \gamma_c \cdot \frac{x_{n,c,s} - \mu_{n,g(c)}}{\sqrt{\sigma^2_{n,g(c)} + \epsilon}} + \beta_c

    Independence from batch size avoids the train/eval mismatch that
    BatchNorm requires running buffers to fix.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import group_norm
    >>> x = lucid.randn(2, 32, 16, 16)
    >>> y = group_norm(x, num_groups=8)
    >>> y.shape
    (2, 32, 16, 16)
    """
    from lucid._factories.creation import ones, zeros

    C = x.shape[1]
    w = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(C, device=x.device, dtype=x.dtype))
    )
    b = (
        _unwrap(bias)
        if bias is not None
        else _unwrap(zeros(C, device=x.device, dtype=x.dtype))
    )
    return _wrap(_C_engine.nn.group_norm(_unwrap(x), w, b, num_groups, eps))


def rms_norm(
    x: Tensor,
    normalized_shape: list[int] | tuple[int, ...],
    weight: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    r"""Root-mean-square layer normalization (Zhang & Sennrich, 2019).

    A simplified variant of :func:`layer_norm` that drops mean
    centering and the additive bias.  Cheaper to compute and shown to
    perform competitively in large language models (LLaMA, T5,
    PaLM, ...), where its lower latency and memory footprint matter at
    scale.

    Parameters
    ----------
    x : Tensor
        Input whose last dimension matches ``normalized_shape``.
    normalized_shape : list of int or tuple of int
        Trailing dims to normalise over.  Typically ``(d,)`` where
        ``d`` is the model hidden size.
    weight : Tensor, optional
        Per-element scale :math:`\gamma`.  Defaults to ones.
    eps : float, optional
        Numerical safety added inside the square root.

    Returns
    -------
    Tensor
        Same shape as ``x``.

    Notes
    -----
    Math:

    .. math::

        \text{RMS}(x) &= \sqrt{\frac{1}{|S|}\sum_{j \in S} x_j^2 + \epsilon} \\
        y &= \gamma \cdot \frac{x}{\text{RMS}(x)}

    The dropped mean-centering step costs LayerNorm one extra reduction
    and a subtraction; RMSNorm trades that for a small amount of
    expressivity (no shift invariance).  In practice the loss is
    negligible at scale.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import rms_norm
    >>> x = lucid.randn(2, 128, 512)
    >>> y = rms_norm(x, normalized_shape=(512,))
    >>> y.shape
    (2, 128, 512)
    """
    from lucid._factories.creation import ones

    C = x.shape[-1]
    w = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(C, device=x.device, dtype=x.dtype))
    )
    return _wrap(_C_engine.nn.rms_norm(_unwrap(x), w, eps))


def instance_norm(
    x: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Instance normalization (Ulyanov, Vedaldi & Lempitsky, 2016).

    Normalises each ``(sample, channel)`` slice against its **own**
    spatial mean and variance — i.e. statistics are reduced only over
    the spatial axes.  Originally introduced for fast neural style
    transfer; widely used wherever per-image contrast should be made
    invariant (image generation, image-to-image translation).

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, *spatial)`` with ``ndim >= 3``.
    running_mean : Tensor or None, optional
        Running per-channel mean of shape ``(C,)``.  Only consulted
        when ``use_input_stats=False``.
    running_var : Tensor or None, optional
        Running per-channel variance of shape ``(C,)``.
    weight : Tensor, optional
        Per-channel scale :math:`\gamma` of shape ``(C,)``.
    bias : Tensor, optional
        Per-channel shift :math:`\beta` of shape ``(C,)``.
    use_input_stats : bool, optional
        When ``True`` (default) statistics come from the current
        instance.  When ``False`` and running buffers are supplied,
        those are used instead.
    momentum : float, optional
        EMA coefficient for the running buffers (when externally
        updated).
    eps : float, optional
        Numerical safety added inside the square root.

    Returns
    -------
    Tensor
        Same shape as ``x``.

    Notes
    -----
    Math (reduction taken only over spatial axes :math:`S`):

    .. math::

        \mu_{n,c} &= \frac{1}{|S|}\sum_{s \in S} x_{n,c,s} \\
        \sigma^2_{n,c} &= \frac{1}{|S|}\sum_{s \in S} (x_{n,c,s} - \mu_{n,c})^2 \\
        y_{n,c,s} &= \gamma_c \cdot \frac{x_{n,c,s} - \mu_{n,c}}{\sqrt{\sigma^2_{n,c} + \epsilon}} + \beta_c

    InstanceNorm equals :func:`group_norm` with ``num_groups == C``.
    The lack of batch coupling also makes it the natural normaliser
    when adapting per-sample statistics matters more than population
    coherence — exactly the case in style transfer.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import instance_norm
    >>> x = lucid.randn(4, 3, 64, 64)
    >>> y = instance_norm(x)
    >>> y.shape
    (4, 3, 64, 64)
    """
    if x.ndim < 3:
        raise ValueError(
            f"instance_norm: expected at least 3-D input (N, C, *spatial), "
            f"got ndim={x.ndim}"
        )
    spatial_dims: list[int] = list(range(2, x.ndim))
    C: int = int(x.shape[1])
    # Channel-broadcast shape (1, C, 1, 1, ...) for affine + running stats.
    bcast_shape: list[int] = [1, C] + [1] * (x.ndim - 2)

    if use_input_stats or running_mean is None or running_var is None:
        # Per-instance (per-(n,c)) statistics.
        mean: Tensor = x.mean(spatial_dims, keepdim=True)
        var: Tensor = x.var(spatial_dims, keepdim=True, correction=0)
    else:
        mean = running_mean.reshape(bcast_shape)
        var = running_var.reshape(bcast_shape)

    y: Tensor = (x - mean) / (var + eps).sqrt()

    if weight is not None:
        y = y * weight.reshape(bcast_shape)
    if bias is not None:
        y = y + bias.reshape(bcast_shape)
    return y


# ── P3 fill: local_response_norm (functional form of LocalResponseNorm) ────


def local_response_norm(
    x: Tensor,
    size: int,
    alpha: float = 1e-4,
    beta: float = 0.75,
    k: float = 1.0,
) -> Tensor:
    r"""Local response normalization (Krizhevsky, Sutskever & Hinton, 2012).

    Implements lateral inhibition: each activation is divided by a
    function of the squared activations of its channel neighbours.
    Introduced with AlexNet and largely superseded by batch / layer
    normalisation, but still used in some classic architectures.

    Parameters
    ----------
    x : Tensor
        Input of shape ``(N, C, *spatial)`` (any number of spatial
        dims).  Inputs with fewer than 2 dimensions are returned
        unchanged.
    size : int
        Number of neighbouring channels (the size of the local window)
        used in the normalisation.
    alpha : float, optional
        Multiplicative scaling factor on the squared-activation sum.
        Default ``1e-4``.
    beta : float, optional
        Exponent applied to the normalising factor.  Default ``0.75``.
    k : float, optional
        Additive constant guarding the division.  Default ``1.0``.

    Returns
    -------
    Tensor
        Same shape as ``x``.

    Notes
    -----
    Math (the neighbourhood :math:`\mathcal{N}(c)` is the ``size``
    channels centred at :math:`c`, zero-padded at boundaries):

    .. math::

        y_{i,c,h,w} = \frac{x_{i,c,h,w}}{\left(k + \alpha \sum_{c' \in \mathcal{N}(c)} x_{i,c',h,w}^2 \right)^\beta}

    Boundary handling uses asymmetric zero padding when ``size`` is
    even: ``(size - 1) // 2`` on the left and ``size // 2`` on the
    right, matching the original AlexNet / cuDNN convention.  LRN
    predates BatchNorm and provides only lateral inhibition (no learnt
    affine), which is why modern architectures usually do not include
    it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import local_response_norm
    >>> x = lucid.randn(1, 96, 27, 27)   # AlexNet conv1 output shape
    >>> y = local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2.0)
    >>> y.shape
    (1, 96, 27, 27)
    """
    xi = _unwrap(x)
    if len(xi.shape) < 2:
        return x

    ndim = len(xi.shape)
    C = int(xi.shape[1])
    # Pad totals to ``size - 1`` so the post-pad unfold yields exactly C
    # windows.  Asymmetric for even ``size`` (LAPACK / cuDNN convention).
    pad_l = (size - 1) // 2
    pad_r = size // 2

    x_sq = _C_engine.mul(xi, xi)
    pad_pairs: list[tuple[int, int]] = (
        [(0, 0)] + [(pad_l, pad_r)] + [(0, 0)] * (ndim - 2)
    )
    x_sq_pad = _C_engine.pad(x_sq, pad_pairs, 0.0)

    spatial_size = 1
    for d in range(2, ndim):
        spatial_size *= int(xi.shape[d])
    flat = _C_engine.reshape(
        x_sq_pad, [int(xi.shape[0]), C + pad_l + pad_r, spatial_size]
    )
    # Transpose to (N, S, C+2h) so unfold_dim slides along the last axis.
    flat_t = _C_engine.permute(flat, [0, 2, 1])
    unf = _C_engine.unfold_dim(flat_t, 2, size, 1)  # (N, S, C, size)
    window_sum = _C_engine.sum(unf, [3], False)  # (N, S, C)
    window_sum_t = _C_engine.permute(window_sum, [0, 2, 1])  # (N, C, S)
    out_shape = list(xi.shape)
    window_sum_rs = _C_engine.reshape(window_sum_t, out_shape)

    k_t = _C_engine.full(out_shape, k, xi.dtype, xi.device)
    alpha_t = _C_engine.full(out_shape, alpha, xi.dtype, xi.device)
    scale = _C_engine.pow_scalar(
        _C_engine.add(k_t, _C_engine.mul(alpha_t, window_sum_rs)), beta
    )
    return _wrap(_C_engine.div(xi, scale))
