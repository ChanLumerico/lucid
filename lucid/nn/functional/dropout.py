"""
nn.functional dropout operations.
All implementations use the C++ engine ops; no numpy.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# SELU affine constants (from the SELU paper)
_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805
_ALPHA_PRIME = -_SELU_ALPHA * _SELU_SCALE  # ≈ -1.7580993408473766


def dropout(
    x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly zero elements with probability p during training."""
    return _wrap(_C_engine.nn.dropout(_unwrap(x), p, training))


def dropout1d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    r"""Channel-wise dropout for 1-D sequence / feature inputs.

    Drops entire 1-D feature maps (channels) of an :math:`(N, C, L)`
    tensor with probability ``p``, scaling the survivors by
    :math:`1/(1-p)` to keep the activation magnitude unbiased.
    Standard per-element :func:`dropout` is statistically weak when
    applied within strongly-correlated feature maps (consecutive
    positions in a sequence are highly correlated), so masking the
    *whole* channel at once provides stronger regularisation in
    1-D CNNs and temporal feature extractors.

    Parameters
    ----------
    x : Tensor
        Input tensor, typically of shape :math:`(N, C, L)`.
    p : float, optional
        Channel-drop probability in :math:`[0, 1)` (default ``0.5``).
    training : bool, optional
        When ``False``, returns ``x`` unchanged — identity at
        inference time (default ``True``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``x``.

    Notes
    -----
    For each batch element :math:`n` and channel :math:`c`, draw
    :math:`m_{n,c} \sim \mathrm{Bernoulli}(1 - p)` independently and
    apply

    .. math::

        y_{n, c, \ell} = \frac{m_{n, c}}{1 - p} \cdot x_{n, c, \ell}.

    The :math:`1/(1-p)` scale ("inverted dropout") preserves
    :math:`\mathbb{E}[y] = \mathbb{E}[x]` so no rescaling is needed
    at inference.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import dropout1d
    >>> x = lucid.ones((2, 4, 5))
    >>> y = dropout1d(x, p=0.5, training=True)  # doctest: +SKIP
    """
    return _wrap(_C_engine.nn.dropoutnd(_unwrap(x), p, training))


def dropout2d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    r"""Channel-wise dropout for 2-D (spatial) feature maps.

    Drops entire 2-D feature maps of an :math:`(N, C, H, W)` tensor
    with probability ``p`` and rescales the survivors by
    :math:`1/(1-p)`.  This is the dropout variant of choice for
    convolutional networks (Tompson et al. 2015 "SpatialDropout"):
    adjacent pixels within a feature map are spatially correlated,
    so elementwise dropout removes very little information.
    Channel-wise masking forces independence between feature maps
    instead.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape :math:`(N, C, H, W)`.
    p : float, optional
        Channel-drop probability in :math:`[0, 1)` (default ``0.5``).
    training : bool, optional
        When ``False``, identity (default ``True``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``x``.

    Notes
    -----
    For each batch element :math:`n` and channel :math:`c`, draw
    :math:`m_{n,c} \sim \mathrm{Bernoulli}(1 - p)` and broadcast
    over the spatial dimensions:

    .. math::

        y_{n, c, h, w} = \frac{m_{n, c}}{1 - p} \cdot x_{n, c, h, w}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import dropout2d
    >>> x = lucid.ones((2, 8, 4, 4))
    >>> y = dropout2d(x, p=0.5, training=True)  # doctest: +SKIP
    """
    return _wrap(_C_engine.nn.dropoutnd(_unwrap(x), p, training))


def dropout3d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    r"""Channel-wise dropout for 3-D (volumetric) feature maps.

    Drops entire 3-D feature maps of an :math:`(N, C, D, H, W)`
    tensor with probability ``p`` and rescales survivors by
    :math:`1/(1-p)`.  Used in 3-D ConvNets for video and volumetric
    medical imaging, where spatio-temporal locality makes per-
    element dropout statistically inefficient (same rationale as
    :func:`dropout2d`).

    Parameters
    ----------
    x : Tensor
        Input tensor of shape :math:`(N, C, D, H, W)`.
    p : float, optional
        Channel-drop probability in :math:`[0, 1)` (default ``0.5``).
    training : bool, optional
        When ``False``, identity (default ``True``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``x``.

    Notes
    -----
    For each batch element :math:`n` and channel :math:`c`, draw
    :math:`m_{n,c} \sim \mathrm{Bernoulli}(1 - p)` and broadcast
    over the volumetric dimensions:

    .. math::

        y_{n, c, d, h, w} = \frac{m_{n, c}}{1 - p} \cdot x_{n, c, d, h, w}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import dropout3d
    >>> x = lucid.ones((1, 4, 3, 4, 4))
    >>> y = dropout3d(x, p=0.5, training=True)  # doctest: +SKIP
    """
    return _wrap(_C_engine.nn.dropoutnd(_unwrap(x), p, training))


def alpha_dropout(
    x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    r"""Alpha dropout — variance-preserving dropout for SELU networks.

    Designed to be paired with the SELU activation (Klambauer
    et al. 2017 "Self-Normalizing Neural Networks").  Standard
    dropout breaks the carefully-tuned zero-mean / unit-variance
    propagation that makes SELU networks self-normalising; alpha
    dropout fixes this by replacing dropped activations with the
    SELU saturation value :math:`\alpha' = -\alpha \cdot \text{scale}`
    (rather than zero), then applying an affine rescaling chosen so
    that the post-dropout activations again have zero mean and unit
    variance.

    Parameters
    ----------
    x : Tensor
        Input tensor, any shape.
    p : float, optional
        Element-drop probability in :math:`[0, 1]` (default ``0.5``).
    training : bool, optional
        When ``False``, identity (default ``True``).
    inplace : bool, optional
        Reserved for API parity; the implementation always
        produces a new tensor (default ``False``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``x``.

    Notes
    -----
    With keep probability :math:`q = 1 - p`, each element is
    independently set to its original value (with probability
    :math:`q`) or to :math:`\alpha'` (with probability :math:`p`).
    The result is then affine-transformed:

    .. math::

        y = a \cdot \mathrm{mix}(x, \alpha') + b,
        \quad
        a = \big(q + p\,(\alpha')^2\,q\big)^{-1/2},
        \quad
        b = -a\,\alpha'\,p,

    chosen so that :math:`\mathbb{E}[y] = 0` and
    :math:`\mathrm{Var}(y) = 1` when the input has zero mean and
    unit variance.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import alpha_dropout
    >>> x = lucid.randn(8)
    >>> y = alpha_dropout(x, p=0.1, training=True)  # doctest: +SKIP
    """
    if not training or p == 0.0:
        return x

    xi = _unwrap(x)

    if p == 1.0:
        return _wrap(_C_engine.zeros(xi.shape, xi.dtype, xi.device))

    keep_prob = 1.0 - p
    a_coeff = (keep_prob + _ALPHA_PRIME**2 * keep_prob * p) ** (-0.5)
    b_coeff = -a_coeff * _ALPHA_PRIME * p

    # Bernoulli mask (1 = keep, 0 = drop) via uniform sampling
    rand_s = _C_engine.rand(xi.shape, xi.dtype, xi.device)
    thresh = _C_engine.full(xi.shape, keep_prob, xi.dtype, xi.device)
    keep = _C_engine.less(rand_s, thresh)  # bool mask: True → keep
    alpha_t = _C_engine.full(xi.shape, _ALPHA_PRIME, xi.dtype, xi.device)
    mixed = _C_engine.where(keep, xi, alpha_t)  # x or alpha' per element

    # Affine: a * mixed + b
    a_t = _C_engine.full(xi.shape, a_coeff, xi.dtype, xi.device)
    b_t = _C_engine.full(xi.shape, b_coeff, xi.dtype, xi.device)
    out = _C_engine.add(_C_engine.mul(a_t, mixed), b_t)
    return _wrap(out)


def feature_alpha_dropout(
    x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    r"""Channel-wise alpha dropout for SELU convolutional networks.

    Combines the channel-granularity of :func:`dropout2d` /
    :func:`dropout3d` with the variance-preserving substitution of
    :func:`alpha_dropout`.  Whole channels are dropped (replaced
    rather than zeroed) and the result is affine-rescaled to keep
    the self-normalising property — appropriate for SELU-activated
    CNNs where spatially-correlated activations would make per-
    element alpha dropout ineffective.

    For inputs of rank less than 2, falls back to per-element
    :func:`alpha_dropout`.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape :math:`(N, C, *)` (one or more
        spatial dimensions).
    p : float, optional
        Channel-drop probability in :math:`[0, 1]` (default ``0.5``).
    training : bool, optional
        When ``False``, identity (default ``True``).
    inplace : bool, optional
        Reserved for API parity; ignored (default ``False``).

    Returns
    -------
    Tensor
        Same shape and dtype as ``x``.

    Notes
    -----
    A Bernoulli mask of shape :math:`(N, C)` is drawn and
    broadcast across the spatial axes; surviving channels are
    multiplied by 1 and dropped channels by 0.  (The full alpha-
    rescaling that :func:`alpha_dropout` applies element-wise is
    handled at the channel level here.)  Variance preservation
    holds in expectation when the input statistics already satisfy
    the SELU fixed-point assumptions.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import feature_alpha_dropout
    >>> x = lucid.randn(2, 4, 3, 3)
    >>> y = feature_alpha_dropout(x, p=0.25, training=True)  # doctest: +SKIP
    """
    if not training or p == 0.0:
        return x

    xi = _unwrap(x)
    if len(xi.shape) < 2:
        return alpha_dropout(x, p, training, inplace)

    N, C = int(xi.shape[0]), int(xi.shape[1])
    keep_prob = 1.0 - p

    # (N, C) Bernoulli mask
    rand_nc = _C_engine.rand([N, C], xi.dtype, xi.device)
    thresh = _C_engine.full([N, C], keep_prob, xi.dtype, xi.device)
    keep_nc = _C_engine.less(rand_nc, thresh)  # (N, C) bool

    # Reshape to (N, C, 1, 1, ...) and broadcast_to input shape
    spatial_dims = len(xi.shape) - 2
    mask_shape = [N, C] + [1] * spatial_dims
    keep_broad = _C_engine.reshape(keep_nc, mask_shape)
    keep_broad = _C_engine.broadcast_to(keep_broad, list(xi.shape))

    zeros_t = _C_engine.zeros(xi.shape, xi.dtype, xi.device)
    out = _C_engine.where(keep_broad, xi, zeros_t)
    return _wrap(out)
