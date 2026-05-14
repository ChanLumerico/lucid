"""
nn.functional activation functions.
"""

from typing import TYPE_CHECKING

import lucid as _l
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def relu(x: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the rectified linear unit function element-wise.

    The simplest and most widely-used activation function in modern deep
    learning.  Zeros out negative inputs and passes positive inputs
    through unchanged — a piecewise-linear sparsifier that gives neural
    networks their nonlinearity without saturating gradients for large
    positive activations.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape.
    inplace : bool, optional
        Reserved for API compatibility — currently a no-op (Lucid always
        allocates a fresh output).

    Returns
    -------
    Tensor
        Element-wise ReLU output, same shape and dtype as ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{ReLU}(x) = \max(0,\, x)

    Gradient is the indicator :math:`\partial \text{ReLU}/\partial x =
    \mathbb{1}_{x > 0}` — zero for negative inputs, one for positive,
    undefined at exactly zero (convention: 0).  This causes the **dying
    ReLU** problem: a neuron that always produces negative pre-activations
    receives no gradient and stops learning.  Variants like
    :func:`leaky_relu`, :func:`elu`, :func:`gelu`, and :func:`silu`
    address this by allowing a small (or smooth) gradient for negative
    inputs.

    Despite its simplicity, ReLU remains a strong default for CNN
    backbones; transformers commonly prefer :func:`gelu` for its smooth
    non-monotonic shape.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import relu
    >>> relu(lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    Tensor([0., 0., 0., 1., 2.])
    """
    return _wrap(_C_engine.relu(_unwrap(x)))


def leaky_relu(
    x: Tensor, negative_slope: float = 0.01, inplace: bool = False
) -> Tensor:
    r"""Leaky rectified linear unit activation.

    A simple modification of :func:`relu` that lets a small, non-zero gradient
    pass through for negative inputs.  This avoids the *dying ReLU* problem
    where neurons stuck on the negative side stop updating because their
    gradient is exactly zero.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is applied element-wise.
    negative_slope : float, optional
        Slope :math:`\alpha` of the linear branch for negative inputs.
        Default ``0.01``.
    inplace : bool, optional
        Accepted for API compatibility; currently ignored.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{LeakyReLU}(x) = \max(0, x) + \alpha \min(0, x)

    The derivative is :math:`1` for :math:`x > 0` and :math:`\alpha` for
    :math:`x \le 0`.  Unlike :func:`relu`, the derivative is everywhere
    non-zero (assuming :math:`\alpha \ne 0`), so gradient signal continues
    to flow through inactive units.  Use Leaky ReLU when ReLU networks are
    showing many permanently dead neurons; for a learnable slope, see
    :func:`prelu`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import leaky_relu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0])
    >>> leaky_relu(x, negative_slope=0.1)
    Tensor([-0.2000, -0.1000,  0.0000,  1.0000])
    """
    return _wrap(_C_engine.leaky_relu(_unwrap(x), negative_slope))


def elu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""Exponential linear unit activation.

    A smooth alternative to :func:`relu` that saturates to :math:`-\alpha` on
    the negative side instead of clipping at zero.  The exponential branch
    pushes the mean activation closer to zero, which speeds up training by
    reducing internal covariate shift (Clevert et al. 2015).

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.
    alpha : float, optional
        Saturation value :math:`\alpha > 0` for negative inputs.
        Default ``1.0``.
    inplace : bool, optional
        Accepted for API compatibility; currently ignored.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{ELU}(x) = \begin{cases}
            x & x > 0 \\
            \alpha (e^x - 1) & x \le 0
        \end{cases}

    Continuous and once-differentiable at the origin (derivative is
    :math:`1` from above and :math:`\alpha` from below — equal when
    :math:`\alpha = 1`).  Unlike ReLU it produces negative outputs, which
    helps push activation means toward zero; unlike Leaky ReLU it bounds
    the negative tail, providing implicit noise robustness.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import elu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0])
    >>> elu(x)
    Tensor([-0.8647, -0.6321,  0.0000,  1.0000])
    """
    return _wrap(_C_engine.elu(_unwrap(x), alpha))


def selu(x: Tensor, inplace: bool = False) -> Tensor:
    r"""Scaled exponential linear unit activation.

    Self-normalising activation from Klambauer et al. (2017): when used in
    a fully-connected network with LeCun-normal initialised weights, the
    fixed scale :math:`\lambda` and slope :math:`\alpha` drive the
    activations toward zero-mean, unit-variance fixed points without the
    need for explicit batch / layer normalisation.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.
    inplace : bool, optional
        Accepted for API compatibility; currently ignored.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{SELU}(x) = \lambda \, \text{ELU}(x, \alpha) =
        \lambda \begin{cases}
            x & x > 0 \\
            \alpha (e^x - 1) & x \le 0
        \end{cases}

    with the (non-tunable) constants

    .. math::

        \lambda \approx 1.0507, \qquad \alpha \approx 1.6733,

    chosen so that the activation has a zero-mean unit-variance attractor
    fixed point.  Pair with ``alpha_dropout`` and LeCun-normal weight init
    to retain the self-normalising property.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import selu
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> selu(x)
    Tensor([-1.1113,  0.0000,  1.0507])
    """
    return _wrap(_C_engine.selu(_unwrap(x)))


def _erf_approx(xi: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """Polynomial approximation of erf (Abramowitz & Stegun 7.1.26, max err < 1.5e-7)."""
    p_coef = 0.3275911
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    shape = list(xi.shape)
    dt, dev = xi.dtype, xi.device

    abs_xi = _C_engine.abs(xi)
    ones = _C_engine.ones(shape, dt, dev)
    # t = 1 / (1 + p * |x|)
    t = _C_engine.div(
        ones,
        _C_engine.add(
            ones, _C_engine.mul(_C_engine.full(shape, p_coef, dt, dev), abs_xi)
        ),
    )
    # Horner: ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t
    poly = _C_engine.add(
        _C_engine.mul(_C_engine.full(shape, a5, dt, dev), t),
        _C_engine.full(shape, a4, dt, dev),
    )
    poly = _C_engine.add(_C_engine.mul(poly, t), _C_engine.full(shape, a3, dt, dev))
    poly = _C_engine.add(_C_engine.mul(poly, t), _C_engine.full(shape, a2, dt, dev))
    poly = _C_engine.add(_C_engine.mul(poly, t), _C_engine.full(shape, a1, dt, dev))
    poly = _C_engine.mul(poly, t)
    # erf(|x|) = 1 - poly * exp(-x^2)
    erf_abs = _C_engine.sub(
        ones,
        _C_engine.mul(
            poly, _C_engine.exp(_C_engine.neg(_C_engine.mul(abs_xi, abs_xi)))
        ),
    )
    # Restore sign: erf(x) = sign(x) * erf(|x|), with erf(0)=0 handled by sign(0)=0
    return _C_engine.where(
        _C_engine.equal(xi, _C_engine.zeros(shape, dt, dev)),
        _C_engine.zeros(shape, dt, dev),
        _C_engine.mul(_C_engine.sign(xi), erf_abs),
    )


def gelu(x: Tensor, approximate: str = "none") -> Tensor:
    r"""Gaussian Error Linear Unit activation.

    Smooth, non-monotonic activation that has largely replaced ReLU in
    transformer architectures.  Approximates :math:`x \cdot \mathbb{1}_{x>0}`
    but is differentiable everywhere and lets a small negative signal pass
    through, which improves gradient flow in deep networks.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.
    approximate : str, optional
        Either ``"none"`` (default, exact formula via ``erf``) or
        ``"tanh"`` (faster polynomial approximation used in BERT /
        Hendrycks 2016).

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    Exact form:

    .. math::

        \text{GELU}(x) = x \, \Phi(x)
        = \frac{x}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]

    where :math:`\Phi` is the standard normal CDF.  The ``"tanh"``
    approximation is

    .. math::

        \text{GELU}(x) \approx \frac{x}{2}\left[1 +
        \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\, x^3)\right)\right].

    Unlike ReLU, GELU has a non-zero gradient everywhere — useful for
    training very deep transformer stacks.  Introduced by
    Hendrycks & Gimpel (2016) and adopted widely after BERT.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import gelu
    >>> x = lucid.tensor([-1.0, 0.0, 1.0, 2.0])
    >>> gelu(x)
    Tensor([-0.1587,  0.0000,  0.8413,  1.9545])
    """
    xi = _unwrap(x)
    if approximate == "tanh":
        return _wrap(_C_engine.gelu(xi))
    # exact: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2_inv = 0.7071067811865476
    shape = list(xi.shape)
    dt, dev = xi.dtype, xi.device
    x_scaled = _C_engine.mul(xi, _C_engine.full(shape, sqrt2_inv, dt, dev))
    erf_val = _erf_approx(x_scaled)
    half = _C_engine.full(shape, 0.5, dt, dev)
    ones = _C_engine.ones(shape, dt, dev)
    return _wrap(_C_engine.mul(_C_engine.mul(xi, half), _C_engine.add(ones, erf_val)))


def silu(x: Tensor, inplace: bool = False) -> Tensor:
    r"""Sigmoid Linear Unit (a.k.a. Swish-1) activation.

    Smooth, non-monotonic activation popularised by Ramachandran et al.
    (2017) as the result of a neural architecture search over activation
    functions.  Equivalent to Swish with :math:`\beta = 1`.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.
    inplace : bool, optional
        Accepted for API compatibility; currently ignored.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{SiLU}(x) = x \, \sigma(x) = \frac{x}{1 + e^{-x}}

    Has a small negative bump near :math:`x \approx -1.28` (minimum value
    :math:`\approx -0.278`); approaches 0 from below for large negative
    input and approaches the identity for large positive input.  The
    derivative is

    .. math::

        \text{SiLU}'(x) = \sigma(x) + x\sigma(x)(1 - \sigma(x)),

    which is smooth and unbounded above.  Often outperforms ReLU in deep
    CNNs and is the default in EfficientNet and many MoE blocks.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import silu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> silu(x)
    Tensor([-0.2384, -0.2689,  0.0000,  0.7311,  1.7616])
    """
    return _wrap(_C_engine.silu(_unwrap(x)))


def mish(x: Tensor) -> Tensor:
    r"""Mish activation function.

    Self-regularising non-monotonic activation introduced by Misra (2019).
    Like :func:`silu` it has a small negative dip and approaches the
    identity for large positive inputs, but its tanh-of-softplus form gives
    slightly smoother gradients which tend to help convergence in CV tasks
    (YOLOv4 uses it throughout).

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{Mish}(x) = x \, \tanh\!\big(\text{softplus}(x)\big)
                      = x \, \tanh\!\big(\ln(1 + e^x)\big)

    :math:`C^\infty`-smooth, non-monotonic, unbounded above and bounded
    below (minimum :math:`\approx -0.3088`).  Empirically outperforms
    ReLU/Swish on some object-detection benchmarks at modest extra cost.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import mish
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> mish(x)
    Tensor([-0.2525, -0.3034,  0.0000,  0.8651,  1.9440])
    """
    return _wrap(_C_engine.mish(_unwrap(x)))


def hardswish(x: Tensor) -> Tensor:
    r"""Hard Swish activation — piecewise linear approximation of :func:`silu`.

    Introduced in MobileNetV3 (Howard et al. 2019) as a compute-friendly
    replacement for the sigmoid in Swish: the exponential is approximated
    by a clipped linear function, which is far cheaper on mobile / quantised
    hardware while preserving the shape of the activation.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{HardSwish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}
        = \begin{cases}
            0 & x \le -3 \\
            x(x+3)/6 & -3 < x < 3 \\
            x & x \ge 3
        \end{cases}

    Continuous but only piecewise-:math:`C^1` (the second derivative jumps
    at :math:`x = \pm 3`).  Cheap to evaluate and to quantise — preferred
    over :func:`silu` in mobile architectures.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hardswish
    >>> x = lucid.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
    >>> hardswish(x)
    Tensor([ 0.0000, -0.3333,  0.0000,  0.6667,  4.0000])
    """
    return _wrap(_C_engine.hard_swish(_unwrap(x)))


def hardsigmoid(x: Tensor) -> Tensor:
    r"""Hard sigmoid — piecewise linear approximation of :func:`sigmoid`.

    Avoids the cost of an exponential by clipping a scaled-and-shifted
    identity to :math:`[0, 1]`.  Widely used as the gating function inside
    quantised mobile networks (MobileNetV3 Squeeze-and-Excitation blocks).

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``, values in
        :math:`[0, 1]`.

    Notes
    -----
    .. math::

        \text{HardSigmoid}(x) = \begin{cases}
            0 & x \le -3 \\
            x/6 + 1/2 & -3 < x < 3 \\
            1 & x \ge 3
        \end{cases}

    Matches :func:`sigmoid` exactly at :math:`x = 0` (both return ``0.5``)
    but is only piecewise-linear.  Derivative is :math:`1/6` in the linear
    region and :math:`0` in the saturated regions.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hardsigmoid
    >>> x = lucid.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
    >>> hardsigmoid(x)
    Tensor([0.0000, 0.3333, 0.5000, 0.6667, 1.0000])
    """
    return _wrap(_C_engine.hard_sigmoid(_unwrap(x)))


def sigmoid(x: Tensor) -> Tensor:
    r"""Logistic sigmoid activation.

    Squashes any real-valued input into the open interval :math:`(0, 1)`.
    Historically the canonical activation for binary classification heads
    and the gating function in LSTM / GRU cells.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``, values in
        :math:`(0, 1)`.

    Notes
    -----
    .. math::

        \sigma(x) = \frac{1}{1 + e^{-x}}

    Derivative :math:`\sigma'(x) = \sigma(x)(1 - \sigma(x)) \le 0.25`.
    The bounded derivative is the classical cause of the vanishing
    gradient problem in deep feed-forward stacks, so for hidden layers
    prefer ReLU / GELU / SiLU.  For binary classification logits, use
    :func:`logsigmoid` to compute :math:`\log\sigma(x)` in a numerically
    stable way.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import sigmoid
    >>> x = lucid.tensor([-2.0, 0.0, 2.0])
    >>> sigmoid(x)
    Tensor([0.1192, 0.5000, 0.8808])
    """
    return _wrap(_C_engine.sigmoid(_unwrap(x)))


def tanh(x: Tensor) -> Tensor:
    r"""Hyperbolic tangent activation.

    A rescaled, recentred sigmoid that maps the real line to
    :math:`(-1, 1)`.  Because outputs are zero-centred, tanh is generally
    preferred over :func:`sigmoid` for hidden layers in shallow networks
    and is the canonical squashing function in RNN cells.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``, values in
        :math:`(-1, 1)`.

    Notes
    -----
    .. math::

        \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
                 = 2\sigma(2x) - 1

    Derivative :math:`1 - \tanh^2(x)`, maximum value :math:`1` at the
    origin.  Saturates for :math:`|x| \gtrsim 3`; in deep networks this
    leads to vanishing gradients, so modern feed-forward stacks prefer
    ReLU-family activations.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import tanh
    >>> x = lucid.tensor([-2.0, 0.0, 2.0])
    >>> tanh(x)
    Tensor([-0.9640,  0.0000,  0.9640])
    """
    return _wrap(_C_engine.tanh(_unwrap(x)))


def softmax(x: Tensor, dim: int | None = None) -> Tensor:
    r"""Apply the softmax function along a dimension.

    Converts a vector of real-valued logits into a probability
    distribution: the outputs are non-negative and sum to one along
    ``dim``.  The central tool for multi-class classification heads and
    attention weights.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape (the "logits").
    dim : int, optional
        Dimension along which softmax is computed.  Defaults to ``-1``
        (the last axis).

    Returns
    -------
    Tensor
        Same-shape tensor whose entries along ``dim`` form a probability
        simplex (each non-negative, summing to 1).

    Notes
    -----
    Mathematical definition (per-vector along ``dim``):

    .. math::

        \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

    A naïve implementation overflows for large positive ``x``; the engine
    uses the standard log-sum-exp shift :math:`x_i \mapsto x_i - \max_j x_j`
    to evaluate it in finite precision.

    For loss computation, prefer :func:`log_softmax` followed by
    :func:`nll_loss` (or :func:`cross_entropy` end-to-end), since the
    composition of ``log(softmax(...))`` loses precision.  The gradient
    has the convenient closed form
    :math:`\partial p_i / \partial x_j = p_i (\delta_{ij} - p_j)` —
    the Jacobian factorises out cleanly during backprop through softmax-CE.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softmax
    >>> logits = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> p = softmax(logits, dim=1)
    >>> p
    Tensor([[0.0900, 0.2447, 0.6652]])
    >>> p.sum(dim=1)
    Tensor([1.0000])
    """
    axis = dim if dim is not None else -1
    return _wrap(_C_engine.softmax(_unwrap(x), axis))


def log_softmax(x: Tensor, dim: int | None = None) -> Tensor:
    r"""Numerically stable log-softmax along a dimension.

    Equivalent to ``log(softmax(x, dim))`` but computed in a way that
    avoids overflow in the exponentials and the loss of precision incurred
    by taking the logarithm of small softmax probabilities.  Almost always
    the right thing to use as the input to negative-log-likelihood / NLL
    classification losses.

    Parameters
    ----------
    x : Tensor
        Input logits of any shape.
    dim : int, optional
        Dimension along which the log-softmax is computed.  Defaults to
        the last dimension (``-1``).

    Returns
    -------
    Tensor
        Log-probabilities of the same shape as ``x``, summing-to-one on
        the exponentiated scale along ``dim``.

    Notes
    -----
    .. math::

        \text{log\_softmax}(x)_i = x_i - \log\!\sum_j e^{x_j}
        = (x_i - m) - \log\!\sum_j e^{x_j - m}, \quad m = \max_j x_j

    The :math:`-m` shift inside the exponent is the key numerical trick:
    it makes every exponent non-positive so :math:`e^{x_j - m} \in (0, 1]`
    and the sum is bounded.  Pairing ``log_softmax`` with
    :func:`nll_loss` is numerically equivalent to — and more stable than —
    ``softmax`` followed by :func:`cross_entropy`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import log_softmax
    >>> logits = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> log_softmax(logits, dim=1)
    Tensor([[-2.4076, -1.4076, -0.4076]])
    """
    axis = dim if dim is not None else -1
    sm = _C_engine.softmax(_unwrap(x), axis)
    return _wrap(_C_engine.log(sm))


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    r"""Smooth approximation of :func:`relu`.

    Softplus is a strictly positive, :math:`C^\infty` smooth function that
    behaves like :math:`x` for large positive inputs and like :math:`e^x`
    for large negative inputs.  Its derivative is :func:`sigmoid`, which is
    why softplus appears naturally as the log-partition of a Bernoulli and
    as the analytic primitive of the logistic function.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.
    beta : float, optional
        Sharpness parameter :math:`\beta > 0`.  Larger ``beta`` makes the
        curve approach a hard ReLU.  Default ``1.0``.
    threshold : float, optional
        For numerical stability the formula collapses to the identity
        :math:`x` whenever :math:`\beta x > \text{threshold}` (the
        :math:`e^{\beta x}` term would otherwise overflow).  Default
        ``20.0``.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``, strictly positive.

    Notes
    -----
    .. math::

        \text{softplus}_\beta(x) = \frac{1}{\beta}\log\!\big(1 + e^{\beta x}\big)

    Derivative is :math:`\sigma(\beta x)`.  The overflow guard at
    ``threshold`` exploits that for large :math:`\beta x` the function is
    indistinguishable from :math:`x` in finite precision.  The engine's
    bare ``softplus`` kernel implements the special case
    :math:`\beta = 1`, :math:`\text{threshold} \to \infty`; when either
    argument deviates the full formula is composed so autograd still
    flows through the existing backward nodes.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softplus
    >>> x = lucid.tensor([-2.0, 0.0, 2.0])
    >>> softplus(x)
    Tensor([0.1269, 0.6931, 2.1269])
    """
    if beta == 1.0 and threshold >= 50.0:
        # Hot-path: the bare engine kernel is already correct here.
        return _wrap(_C_engine.softplus(_unwrap(x)))
    xi = _unwrap(x)
    beta_t = _C_engine.full(list(xi.shape), float(beta), xi.dtype, xi.device)
    threshold_t = _C_engine.full(list(xi.shape), float(threshold), xi.dtype, xi.device)
    bx = _C_engine.mul(xi, beta_t)
    # log1p(exp(bx)) / beta — composed because the engine's ``softplus`` is
    # beta-agnostic.
    inv_beta = _C_engine.full(list(xi.shape), 1.0 / float(beta), xi.dtype, xi.device)
    softplus_full = _C_engine.mul(_C_engine.softplus(bx), inv_beta)
    above = _C_engine.greater(bx, threshold_t)
    return _wrap(_C_engine.where(above, xi, softplus_full))


def relu6(x: Tensor, inplace: bool = False) -> Tensor:
    r"""Rectified linear unit clipped at six.

    Introduced for quantisation-friendly mobile architectures
    (MobileNetV1 / V2): bounding ReLU above keeps activations inside a
    small fixed range so that low-precision integer arithmetic does not
    overflow.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.
    inplace : bool, optional
        Accepted for API compatibility; currently ignored.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``, values in
        :math:`[0, 6]`.

    Notes
    -----
    .. math::

        \text{ReLU6}(x) = \min(\max(0, x), 6)

    Derivative is :math:`1` on :math:`(0, 6)` and :math:`0` outside —
    so beyond suffering from "dead neurons" on the negative side, ReLU6
    also kills gradients above :math:`6`.  The trade-off is worthwhile
    for int8 quantisation where the symmetric :math:`[0, 6]` range maps
    cleanly to a small dynamic-range integer codebook.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import relu6
    >>> x = lucid.tensor([-1.0, 0.0, 3.0, 7.0])
    >>> relu6(x)
    Tensor([0.0000, 0.0000, 3.0000, 6.0000])
    """
    return _wrap(_C_engine.relu6(_unwrap(x)))


def softmin(x: Tensor, dim: int | None = None) -> Tensor:
    r"""Softmin — softmax applied to the negation of the input.

    Useful when small input values should receive high probability, e.g.
    when ``x`` represents distances or costs and a soft-argmin is needed.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape.
    dim : int, optional
        Dimension along which softmin is computed.  Defaults to the last
        dimension (``-1``).

    Returns
    -------
    Tensor
        Tensor of the same shape with values summing to ``1`` along
        ``dim``.

    Notes
    -----
    .. math::

        \text{softmin}(x)_i = \text{softmax}(-x)_i = \frac{e^{-x_i}}{\sum_j e^{-x_j}}

    Exactly equivalent to :func:`softmax` on the negated input; produced
    as a separate function purely for readability when modelling
    minimum-cost attention or temperature-scaled retrieval.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softmin
    >>> x = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> softmin(x, dim=1)
    Tensor([[0.6652, 0.2447, 0.0900]])
    """
    axis = dim if dim is not None else -1
    return _wrap(_C_engine.softmax(_C_engine.neg(_unwrap(x)), axis))


def glu(x: Tensor, dim: int = -1) -> Tensor:
    r"""Gated Linear Unit (Dauphin et al. 2017).

    Splits the input in half along ``dim`` and multiplies the first half
    element-wise by the sigmoid of the second.  Acts as a learnable gate
    that lets the network modulate information flow per channel; widely
    used in sequence models and Conformer-style speech architectures.

    Parameters
    ----------
    x : Tensor
        Input tensor.  The size along ``dim`` must be even — it is split
        into two halves :math:`a` and :math:`b` of equal size.
    dim : int, optional
        Dimension along which to split.  Default ``-1``.

    Returns
    -------
    Tensor
        Output tensor whose shape matches ``x`` except along ``dim``,
        where the size is halved.

    Notes
    -----
    .. math::

        \text{GLU}(x) = a \odot \sigma(b),
        \qquad x = [a; b] \text{ split along } \text{dim}

    The sigmoid gate :math:`\sigma(b)` selects how much of :math:`a` to
    pass through.  Because the gate is multiplicative the derivative is
    well-behaved (no saturation in :math:`a`'s direction), which helps
    train deeper stacks than a plain feed-forward layer would allow.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import glu
    >>> x = lucid.tensor([[1.0, 2.0, 0.0, 1.0]])  # split → a=[1,2], b=[0,1]
    >>> glu(x, dim=-1)
    Tensor([[0.5000, 1.4621]])
    """
    impl = _unwrap(x)
    n = impl.shape[dim] // 2
    # Split into two halves along dim
    parts = _C_engine.split_at(impl, [n], dim)
    first, second = parts[0], parts[1]
    return _wrap(_C_engine.mul(first, _C_engine.sigmoid(second)))


def prelu(x: Tensor, weight: Tensor) -> Tensor:
    r"""Parametric Rectified Linear Unit (He et al. 2015).

    Variant of :func:`leaky_relu` whose negative slope is a **learnable**
    parameter rather than a fixed hyperparameter.  In practice ``weight``
    may be a scalar (shared across channels) or a per-channel vector,
    matching the convention of the reference framework's ``nn.PReLU``.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    weight : Tensor
        Learnable slope :math:`\alpha`.  Either a scalar tensor (slope
        shared across all elements) or a 1-D tensor broadcastable against
        ``x`` along the channel dimension.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{PReLU}(x) = \max(0, x) + \alpha \min(0, x)

    Allows the optimiser to discover the *optimal* negative-side slope
    per channel; the original paper showed this reaches super-human
    ImageNet accuracy.  When :math:`\alpha = 0` recovers :func:`relu`; for
    a fixed non-zero :math:`\alpha` see :func:`leaky_relu`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import prelu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0])
    >>> w = lucid.tensor(0.25)
    >>> prelu(x, w)
    Tensor([-0.5000, -0.2500,  0.0000,  1.0000])
    """
    xi = _unwrap(x)
    wi = _unwrap(weight)
    pos = _C_engine.relu(xi)
    neg_part = _C_engine.mul(
        wi, _C_engine.minimum(_C_engine.zeros(xi.shape, xi.dtype, xi.device), xi)
    )
    return _wrap(_C_engine.add(pos, neg_part))


def celu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""Continuously-differentiable Exponential Linear Unit (Barron 2017).

    A reparameterisation of :func:`elu` whose first derivative is
    continuous at the origin for *every* :math:`\alpha > 0`, not only the
    specific value :math:`\alpha = 1`.  This makes CELU friendlier to
    optimisers that exploit smooth gradients.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.
    alpha : float, optional
        Saturation value :math:`\alpha > 0` for negative inputs.
        Default ``1.0``.
    inplace : bool, optional
        Accepted for API compatibility; currently ignored.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{CELU}(x) = \max(0, x) + \min\!\big(0, \alpha (e^{x/\alpha} - 1)\big)

    At the origin both branches have derivative :math:`1` regardless of
    :math:`\alpha`, removing the kink ELU has when :math:`\alpha \ne 1`.
    Reduces to ELU when :math:`\alpha = 1`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import celu
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0])
    >>> celu(x, alpha=1.0)
    Tensor([-0.8647, -0.6321,  0.0000,  1.0000])
    """
    xi = _unwrap(x)
    pos = _C_engine.relu(xi)
    exp_part = _C_engine.sub(
        _C_engine.exp(
            _C_engine.div(xi, _C_engine.full(xi.shape, alpha, xi.dtype, xi.device))
        ),
        _C_engine.full(xi.shape, 1.0, xi.dtype, xi.device),
    )
    neg = _C_engine.minimum(
        _C_engine.zeros(xi.shape, xi.dtype, xi.device),
        _C_engine.mul(_C_engine.full(xi.shape, alpha, xi.dtype, xi.device), exp_part),
    )
    return _wrap(_C_engine.add(pos, neg))


def hardshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    r"""Hard shrinkage operator.

    Sets elements with magnitude below :math:`\lambda` to zero, leaves
    larger-magnitude elements untouched.  Standard tool for sparse coding
    and the proximal operator of the :math:`\ell_0` "pseudonorm" on a
    coefficient-by-coefficient basis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    lambd : float, optional
        Threshold :math:`\lambda \ge 0`.  Default ``0.5``.

    Returns
    -------
    Tensor
        Shrunk tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{HardShrink}(x) = \begin{cases}
            x & |x| > \lambda \\
            0 & \text{otherwise}
        \end{cases}

    Discontinuous at :math:`x = \pm\lambda` — for an everywhere-continuous
    alternative use :func:`softshrink` (the proximal operator of the
    :math:`\ell_1` norm).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hardshrink
    >>> x = lucid.tensor([-1.0, -0.3, 0.0, 0.3, 1.0])
    >>> hardshrink(x, lambd=0.5)
    Tensor([-1.0000,  0.0000,  0.0000,  0.0000,  1.0000])
    """
    xi = _unwrap(x)
    lam = _C_engine.full(xi.shape, lambd, xi.dtype, xi.device)
    neg_lam = _C_engine.full(xi.shape, -lambd, xi.dtype, xi.device)
    mask = _C_engine.bitwise_or(
        _C_engine.greater(xi, lam),
        _C_engine.less(xi, neg_lam),
    )
    return _wrap(
        _C_engine.where(mask, xi, _C_engine.zeros(xi.shape, xi.dtype, xi.device))
    )


def tanhshrink(x: Tensor) -> Tensor:
    r"""Tanh shrinkage activation.

    Behaves like a smooth high-pass filter: subtracts the tanh-saturated
    component of the input, leaving small linear residuals near zero and
    near-linear behaviour for large magnitudes.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{TanhShrink}(x) = x - \tanh(x)

    For :math:`|x| \ll 1`, :math:`\tanh(x) \approx x - x^3/3`, so
    :math:`\text{TanhShrink}(x) \approx x^3/3` — small signals are
    cubically suppressed.  For :math:`|x| \gg 1`, :math:`\tanh` saturates
    to :math:`\pm 1` so the output behaves like :math:`x \mp 1`.
    :math:`C^\infty`-smooth and monotonic.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import tanhshrink
    >>> x = lucid.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    >>> tanhshrink(x)
    Tensor([-1.0360, -0.0379,  0.0000,  0.0379,  1.0360])
    """
    xi = _unwrap(x)
    return _wrap(_C_engine.sub(xi, _C_engine.tanh(xi)))


def normalize(
    x: Tensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
) -> Tensor:
    r"""Normalize a tensor to unit :math:`L_p` norm along a dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    p : float, optional
        Exponent of the norm (default: 2.0 for Euclidean norm).
    dim : int, optional
        Dimension along which to normalize (default: 1).
    eps : float, optional
        Small value added to the denominator for numerical stability.

    Returns
    -------
    Tensor
        Normalized tensor with unit :math:`L_p` norm along ``dim``.

    Examples
    --------
    >>> x = lucid.tensor([[3.0, 4.0]])
    >>> F.normalize(x, p=2, dim=1)
    tensor([[0.6, 0.8]])

    Notes
    -----
    .. math::

        \hat{x}_i = \frac{x_i}{\max\!\big(\|x\|_p,\, \epsilon\big)},
        \qquad \|x\|_p = \Big(\sum_i |x_i|^p\Big)^{1/p}

    along the chosen dimension.  The ``eps`` clamp avoids division by
    zero when the slice is all zeros.  Use ``p=2`` for cosine-similarity
    style embeddings, ``p=1`` for probability-simplex projection-like
    behaviour, and ``p=inf`` for max-norm clipping.
    """
    return _wrap(_C_engine.nn.lp_normalize(_unwrap(x), p, dim, eps))


def cosine_similarity(
    x1: Tensor,
    x2: Tensor,
    dim: int = 1,
    eps: float = 1e-8,
) -> Tensor:
    r"""Cosine similarity between two tensors along a dimension.

    Measures the angle between two vectors irrespective of their
    magnitudes — the natural similarity metric for unit-norm embedding
    spaces (contrastive learning, retrieval, face verification).

    Parameters
    ----------
    x1, x2 : Tensor
        Tensors of the same shape.
    dim : int, optional
        Dimension along which the dot product / norms are taken.
        Default ``1``.
    eps : float, optional
        Lower bound on the denominator to avoid division by zero.
        Default ``1e-8``.

    Returns
    -------
    Tensor
        Similarity tensor whose shape matches the inputs with ``dim``
        reduced, values in :math:`[-1, 1]`.

    Notes
    -----
    .. math::

        \text{cosine\_sim}(x_1, x_2) =
        \frac{x_1 \cdot x_2}{\max(\|x_1\|_2 \, \|x_2\|_2,\, \epsilon)}

    Internally each input is :math:`L_2`-normalised along ``dim`` via
    :func:`normalize`, after which the dot product is exactly the cosine.
    For pairwise similarity matrices, use ``unsqueeze`` to broadcast the
    inputs into rank-3 tensors before calling this function.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import cosine_similarity
    >>> a = lucid.tensor([[1.0, 0.0]])
    >>> b = lucid.tensor([[1.0, 1.0]])
    >>> cosine_similarity(a, b, dim=1)
    Tensor([0.7071])
    """
    x1n = normalize(x1, p=2.0, dim=dim, eps=eps)
    x2n = normalize(x2, p=2.0, dim=dim, eps=eps)
    # Element-wise product then sum along dim
    impl = _C_engine.sum(_C_engine.mul(_unwrap(x1n), _unwrap(x2n)), [dim])
    return _wrap(impl)


def pairwise_distance(
    x1: Tensor,
    x2: Tensor,
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False,
) -> Tensor:
    r"""Element-wise :math:`L_p` distance between two equally-shaped tensors.

    Computes the per-row :math:`L_p` norm of the difference — the standard
    metric in metric-learning triplet and contrastive losses.

    Parameters
    ----------
    x1, x2 : Tensor
        Tensors of the same shape.  Distance is reduced over the **last**
        dimension.
    p : float, optional
        Order of the norm.  ``2`` for Euclidean, ``1`` for Manhattan.
        Default ``2.0``.
    eps : float, optional
        Small constant added to ``|x1 - x2|`` before the power so the
        derivative is finite even at coincident points.  Default ``1e-6``.
    keepdim : bool, optional
        If ``True``, retain the reduced dimension with size ``1``.
        Default ``False``.

    Returns
    -------
    Tensor
        Distance tensor.

    Notes
    -----
    .. math::

        d_p(x_1, x_2) = \big\| |x_1 - x_2| + \epsilon \big\|_p
        = \Big(\sum_i (|x_{1,i} - x_{2,i}| + \epsilon)^p\Big)^{1/p}

    The ``eps`` shift inside the norm matters for gradient stability — at
    :math:`x_1 = x_2` the bare :math:`L_p` distance is non-differentiable
    when :math:`p < 2`; the offset converts the kink into a finite-slope
    smooth point.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pairwise_distance
    >>> a = lucid.tensor([[1.0, 2.0]])
    >>> b = lucid.tensor([[4.0, 6.0]])
    >>> pairwise_distance(a, b, p=2.0)
    Tensor([5.0000])
    """
    diff = _C_engine.sub(_unwrap(x1), _unwrap(x2))
    # |diff|_p = (sum |diff|^p)^(1/p)
    abs_diff = _C_engine.abs(diff)
    powered = _C_engine.pow(
        _C_engine.add(
            abs_diff,
            _C_engine.full(abs_diff.shape, eps, abs_diff.dtype, abs_diff.device),
        ),
        _C_engine.full(abs_diff.shape, p, abs_diff.dtype, abs_diff.device),
    )
    ndim = len(powered.shape)
    last_dim = ndim - 1
    s = _C_engine.sum(powered, [last_dim], False)
    inv_p = _C_engine.full(s.shape, 1.0 / p, s.dtype, s.device)
    dist = _C_engine.pow(s, inv_p)
    if keepdim:
        dist = _C_engine.unsqueeze(dist, last_dim)
    return _wrap(dist)


def softshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    r"""Soft shrinkage operator — proximal operator of :math:`\lambda \|\cdot\|_1`.

    Translates the input toward zero by :math:`\lambda` and zeros out
    everything in the dead zone :math:`[-\lambda, \lambda]`.  Used in
    iterative-shrinkage solvers (ISTA / FISTA) and in sparse autoencoder
    decoders.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    lambd : float, optional
        Shrinkage threshold :math:`\lambda \ge 0`.  Default ``0.5``.

    Returns
    -------
    Tensor
        Shrunk tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{SoftShrink}(x) = \begin{cases}
            x - \lambda & x > \lambda \\
            x + \lambda & x < -\lambda \\
            0 & \text{otherwise}
        \end{cases}

    Continuous everywhere (unlike :func:`hardshrink`), with derivative
    :math:`1` outside the dead zone and :math:`0` inside it.  Coincides
    with the closed-form solution of
    :math:`\arg\min_y \tfrac12 (y - x)^2 + \lambda |y|`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softshrink
    >>> x = lucid.tensor([-1.0, -0.3, 0.0, 0.3, 1.0])
    >>> softshrink(x, lambd=0.5)
    Tensor([-0.5000,  0.0000,  0.0000,  0.0000,  0.5000])
    """
    xi = _unwrap(x)
    lam = _C_engine.full(xi.shape, lambd, xi.dtype, xi.device)
    neg_lam = _C_engine.full(xi.shape, -lambd, xi.dtype, xi.device)
    zeros = _C_engine.zeros(xi.shape, xi.dtype, xi.device)
    pos_mask = _C_engine.greater(xi, lam)
    neg_mask = _C_engine.less(xi, neg_lam)
    pos_val = _C_engine.sub(xi, lam)
    neg_val = _C_engine.add(xi, lam)
    out = _C_engine.where(pos_mask, pos_val, zeros)
    out = _C_engine.where(neg_mask, neg_val, out)
    return _wrap(out)


# ── P3 fills: missing activations ──────────────────────────────────────────


def hardtanh(
    x: Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
) -> Tensor:
    r"""Hardtanh — element-wise clamp to :math:`[\text{min\_val}, \text{max\_val}]`.

    A cheap piecewise-linear surrogate for :func:`tanh` whose forward and
    backward passes require no transcendental functions.  Standard
    activation in compact / quantised RNN cells.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape.
    min_val : float, optional
        Lower clamp.  Default ``-1.0``.
    max_val : float, optional
        Upper clamp.  Default ``1.0``.
    inplace : bool, optional
        Accepted for API compatibility; currently ignored.

    Returns
    -------
    Tensor
        Clamped tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & x > \text{max\_val} \\
            x & \text{min\_val} \le x \le \text{max\_val} \\
            \text{min\_val} & x < \text{min\_val}
        \end{cases}

    Derivative is :math:`1` inside the linear region and :math:`0` in the
    saturated regions — gradients stop flowing through saturated units, a
    drawback compared to smooth tanh but acceptable when the linear region
    covers the bulk of activations.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hardtanh
    >>> x = lucid.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    >>> hardtanh(x)
    Tensor([-1.0000, -0.5000,  0.0000,  0.5000,  1.0000])
    """
    return _l.clamp(x, min_val, max_val)


def logsigmoid(x: Tensor) -> Tensor:
    r"""Numerically stable :math:`\log \sigma(x)`.

    Direct evaluation of :math:`\log(1 / (1 + e^{-x}))` underflows once
    :math:`\sigma(x)` is below machine epsilon (around :math:`x \lesssim -37`
    for float32).  Rewriting as :math:`-\operatorname{softplus}(-x)`
    matches the same value while staying finite for arbitrarily negative
    inputs.

    Parameters
    ----------
    x : Tensor
        Input logits.

    Returns
    -------
    Tensor
        :math:`\log \sigma(x)` with the same shape as ``x``, values in
        :math:`(-\infty, 0]`.

    Notes
    -----
    .. math::

        \log\sigma(x) = -\log(1 + e^{-x}) = -\operatorname{softplus}(-x)

    Frequently the building block of binary cross-entropy with logits
    (BCE-with-logits) and of importance-weighting computations in RL.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import logsigmoid
    >>> x = lucid.tensor([-10.0, 0.0, 10.0])
    >>> logsigmoid(x)
    Tensor([-10.0000,  -0.6931,  -0.0000])
    """
    return -softplus(-x)


def softsign(x: Tensor) -> Tensor:
    r"""Softsign activation.

    A bounded :math:`(-1, 1)` activation that saturates **polynomially**
    rather than exponentially — its tails are much heavier than
    :func:`tanh`'s, so gradients vanish more slowly in deep networks.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``, values in
        :math:`(-1, 1)`.

    Notes
    -----
    .. math::

        \text{softsign}(x) = \frac{x}{1 + |x|}, \qquad
        \text{softsign}'(x) = \frac{1}{(1 + |x|)^2}

    Derivative decays like :math:`1/x^2` rather than :math:`\tanh`'s
    :math:`1/\cosh^2`, so saturation is gentler.  Lacks the bi-Lipschitz
    smoothness of tanh near zero (kink at :math:`x = 0` from the absolute
    value) but is monotone and zero-centred.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softsign
    >>> x = lucid.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> softsign(x)
    Tensor([-0.6667, -0.5000,  0.0000,  0.5000,  0.6667])
    """
    return x / (1.0 + _l.abs(x))


def threshold(
    x: Tensor,
    threshold: float,
    value: float,
    inplace: bool = False,
) -> Tensor:
    r"""Threshold activation — gate elements by a scalar cutoff.

    Replaces every element of ``x`` that is **not** strictly greater than
    ``threshold`` with the constant ``value``.  A generalisation of
    :func:`relu` (recovered by ``threshold=0, value=0``) and a common
    component of dead-zone non-linearities.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    threshold : float
        Cutoff :math:`t`.  Elements satisfying :math:`x > t` are kept.
    value : float
        Replacement constant for elements not above the threshold.
    inplace : bool, optional
        Accepted for API compatibility; currently ignored.

    Returns
    -------
    Tensor
        Thresholded tensor with the same shape as ``x``.

    Notes
    -----
    .. math::

        \text{threshold}(x; t, v) = \begin{cases}
            x & x > t \\
            v & \text{otherwise}
        \end{cases}

    Discontinuous at :math:`x = t` when :math:`v \ne t`; the derivative
    is :math:`1` above the threshold and :math:`0` below.  Useful as a
    deterministic alternative to :func:`relu` when a non-zero "off"
    value is needed (e.g. setting masked positions to a sentinel).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import threshold
    >>> x = lucid.tensor([-1.0, 0.0, 0.5, 1.0])
    >>> threshold(x, threshold=0.5, value=-99.0)
    Tensor([-99.0000, -99.0000, -99.0000,   1.0000])
    """
    keep = x > threshold
    replacement = _l.full_like(x, float(value))
    return _l.where(keep, x, replacement)


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
) -> Tensor:
    r"""Gumbel-Softmax — differentiable relaxation of categorical sampling.

    Draws Gumbel noise, adds it to the logits, and softmaxes by
    temperature :math:`\tau`.  Lets gradients flow through what would
    otherwise be a non-differentiable categorical sample — central to
    differentiable discrete-action RL, VQ-VAE codebook training, and
    Concrete distributions (Jang et al. / Maddison et al. 2017).

    Parameters
    ----------
    logits : Tensor
        Unnormalised log-probabilities of any shape; the last (or
        ``dim``-th) axis indexes the categorical alternatives.
    tau : float, optional
        Temperature :math:`\tau > 0`.  Smaller :math:`\tau` makes the
        relaxation closer to a one-hot, at the cost of larger gradient
        variance.  Default ``1.0``.
    hard : bool, optional
        If ``True`` the forward pass returns a true one-hot vector while
        the backward pass uses the soft gradient (straight-through
        estimator).  Default ``False``.
    dim : int, optional
        Dimension over which to take the softmax.  Default ``-1``.

    Returns
    -------
    Tensor
        Sample with the same shape as ``logits``.  Rows sum to ``1``
        along ``dim``; in hard mode each row is one-hot.

    Notes
    -----
    .. math::

        g_i \sim \text{Gumbel}(0, 1), \qquad
        y_i = \frac{\exp((\log p_i + g_i)/\tau)}{\sum_j \exp((\log p_j + g_j)/\tau)}

    Gumbel samples are drawn as :math:`-\log(-\log U)` with
    :math:`U \sim \mathrm{Uniform}(0,1)`, clamped away from the boundaries
    to avoid :math:`\log 0`.  In *hard* mode the result is
    :math:`y_{\text{hard}} - y_{\text{soft}}.\text{detach}() + y_{\text{soft}}`,
    which evaluates to :math:`y_{\text{hard}}` in the forward and to
    :math:`y_{\text{soft}}` in the backward.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import gumbel_softmax
    >>> logits = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> y = gumbel_softmax(logits, tau=0.5, hard=True)
    >>> y.sum().item()
    1.0
    """
    # Sample Gumbel noise: −log(−log U) with U ∈ (0, 1).  Clamp away
    # from the boundaries to avoid log(0).
    u: Tensor = _l.rand(
        *tuple(int(s) for s in logits.shape),
        dtype=logits.dtype,
        device=logits.device,
    ).clip(1e-7, 1.0 - 1e-7)
    gumbel: Tensor = -(-(u.log())).log()

    y_soft: Tensor = softmax((logits + gumbel) / tau, dim=dim)

    if not hard:
        return y_soft

    # Straight-through: build a one-hot at argmax, then re-attach
    # ``y_soft``'s gradient via ``y_hard − y_soft.detach() + y_soft``.
    idx: Tensor = y_soft.argmax(dim=dim, keepdim=True)
    y_hard: Tensor = _l.zeros_like(y_soft)
    y_hard = y_hard.scatter(dim, idx, _l.ones_like(idx, dtype=y_soft.dtype))
    return y_hard - y_soft.detach() + y_soft


def rrelu(
    x: Tensor,
    lower: float = 1.0 / 8.0,
    upper: float = 1.0 / 3.0,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    r"""Randomised leaky ReLU (Xu et al. 2015).

    A regularised variant of :func:`leaky_relu`: during **training** every
    negative element is scaled by an i.i.d. uniform slope, injecting
    activation noise on the negative side.  At **evaluation** time the
    expectation of that uniform draw is used, so the function becomes
    deterministic.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    lower : float, optional
        Lower bound of the uniform slope distribution.  Default ``1/8``.
    upper : float, optional
        Upper bound of the uniform slope distribution.  Default ``1/3``.
    training : bool, optional
        If ``True`` sample a fresh slope per element; if ``False`` use the
        midpoint.  Default ``False``.
    inplace : bool, optional
        Accepted for API compatibility; currently ignored.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``x``.

    Notes
    -----
    During training, with :math:`a \sim \mathrm{Uniform}(\text{lower},
    \text{upper})` sampled element-wise:

    .. math::

        \text{RReLU}(x) = \begin{cases}
            x & x \ge 0 \\
            a x & x < 0
        \end{cases}

    During evaluation, :math:`a` is replaced by the expectation
    :math:`(\text{lower} + \text{upper}) / 2`.  The slope randomisation
    acts as a mild noise regulariser — comparable in spirit to dropout
    but applied to activation slope rather than activation magnitude.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import rrelu
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> rrelu(x, training=False)
    Tensor([-0.2292,  0.0000,  1.0000])
    """
    if training:
        # Per-element uniform slope, only applied where x < 0.
        slope: Tensor = (
            _l.rand(*tuple(int(s) for s in x.shape), dtype=x.dtype, device=x.device)
            * (upper - lower)
            + lower
        )
    else:
        # Constant midpoint slope.
        mid: float = 0.5 * (lower + upper)
        slope = _l.full_like(x, mid)
    neg_part: Tensor = slope * x
    return _l.where(x >= 0, x, neg_part)


# ── Inplace variants ──────────────────────────────────────────────────────────
# For ops whose engine already has a native inplace kernel, call it directly.
# For the rest, apply the out-of-place version and copy the result back.


def relu_(x: Tensor, inplace: bool = True) -> Tensor:
    """In-place ReLU via the engine's native inplace kernel."""
    return _wrap(_C_engine.relu_(_unwrap(x)))


def elu_(x: Tensor, alpha: float = 1.0, inplace: bool = True) -> Tensor:
    """In-place ELU: applies ``elu(x, alpha)`` and writes back into ``x``."""
    return x.copy_(elu(x, alpha))


def selu_(x: Tensor, inplace: bool = True) -> Tensor:
    """In-place SELU."""
    return x.copy_(selu(x))


def leaky_relu_(
    x: Tensor, negative_slope: float = 0.01, inplace: bool = True
) -> Tensor:
    """In-place leaky ReLU."""
    return x.copy_(leaky_relu(x, negative_slope))


def hardtanh_(
    x: Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = True,
) -> Tensor:
    """In-place hardtanh."""
    return x.copy_(hardtanh(x, min_val, max_val))


def threshold_(
    x: Tensor,
    threshold_val: float,
    value: float,
    inplace: bool = True,
) -> Tensor:
    """In-place threshold."""
    return x.copy_(threshold(x, threshold_val, value))


def rrelu_(
    x: Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
    inplace: bool = True,
) -> Tensor:
    """In-place RReLU."""
    return x.copy_(rrelu(x, lower, upper, training))
