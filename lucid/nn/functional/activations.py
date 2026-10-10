"""
nn.functional activation functions.
"""

from typing import TYPE_CHECKING

import lucid as _l
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor._metal_views import _check as _metal_check
from lucid._tensor._metal_views import write_back as _metal_write_back
from lucid._types import GeluApproximate
from lucid.nn.functional._inplace import _in_place

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
        If ``True``, write the result into ``x`` and return ``x``.

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
    tensor([0., 0., 0., 1., 2.])
    """
    if not inplace:
        return _wrap(_C_engine.relu(_unwrap(x)))
    if x.requires_grad and _C_engine.grad_enabled():
        return _in_place(x, "relu_", relu)
    # No graph to extend: the engine's fused kernel allocates nothing (0.44 ms
    # against 1.51 ms through the graph-preserving path on 64x3x128x128).
    _metal_check(x)
    _C_engine.relu_(_unwrap(x))
    _metal_write_back(x)
    return x


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
        If ``True``, write the result into ``x`` and return ``x``.

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
    tensor([-0.2, -0.1, 0., 1.])
    """
    if inplace:
        return _in_place(x, "leaky_relu_", lambda t: leaky_relu(t, negative_slope))
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
        If ``True``, write the result into ``x`` and return ``x``.

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
    tensor([-0.8647, -0.6321, 0., 1.])
    """
    if inplace:
        return _in_place(x, "elu_", lambda t: elu(t, alpha))
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
        If ``True``, write the result into ``x`` and return ``x``.

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
    tensor([-1.111, 0., 1.051])
    """
    if inplace:
        return _in_place(x, "selu_", selu)
    return _wrap(_C_engine.selu(_unwrap(x)))


def gelu(x: Tensor, approximate: GeluApproximate = "none") -> Tensor:
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
    tensor([-0.1587, 0., 0.8413, 1.954])
    """
    xi = _unwrap(x)
    if approximate == "tanh":
        return _wrap(_C_engine.gelu(xi))
    return _wrap(_C_engine.gelu_exact(xi))


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
        If ``True``, write the result into ``x`` and return ``x``.

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
    tensor([-0.2384, -0.2689, 0., 0.7311, 1.762])
    """
    if inplace:
        return _in_place(x, "silu_", silu)
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
    tensor([-0.2525, -0.3034, 0., 0.8651, 1.944])
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
    tensor([-0., -0.3333, 0., 0.6667, 4.])
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
    tensor([0., 0.3333, 0.5, 0.6667, 1.])
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
    tensor([0.1192, 0.5, 0.8808])
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
    tensor([-0.964, 0., 0.964])
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
    tensor([[0.09003, 0.2447, 0.6652]])
    >>> p.sum(dim=1)
    tensor([1.])
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
    tensor([[-2.408, -1.408, -0.4076]])
    """
    axis = dim if dim is not None else -1
    # Route through the engine's dedicated log_softmax kernel, which computes
    # ``(x - m) - log(sum(exp(x - m)))`` with the max-subtraction stabilisation.
    # The naive ``log(softmax(x))`` form underflows: at large logits the
    # non-max softmax probabilities round to 0, so ``log(0) = -inf`` (the engine
    # softmax itself is stable, but the subsequent ``log`` is not).
    return _wrap(_C_engine.log_softmax(_unwrap(x), axis))


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
    tensor([0.1269, 0.6931, 2.127])
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
        If ``True``, write the result into ``x`` and return ``x``.

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
    tensor([0., 0., 3., 6.])
    """
    if inplace:
        return _in_place(x, "relu6_", relu6)
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
    tensor([[0.6652, 0.2447, 0.09003]])
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
    tensor([[0.5, 1.462]])
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
    tensor([-0.5, -0.25, 0., 1.])
    """
    xi = _unwrap(x)
    wi = _unwrap(weight)
    # A per-channel ``(C,)`` slope belongs to dim 1 of an ``(N, C, *)`` input.
    # Left as a bare vector it would broadcast against the *last* dim instead,
    # which fails on a feature map (or mixes up channels when that dim is C).
    if len(wi.shape) == 1 and wi.shape[0] > 1 and len(xi.shape) > 2:
        wi = _C_engine.reshape(wi, [wi.shape[0]] + [1] * (len(xi.shape) - 2))
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
        If ``True``, write the result into ``x`` and return ``x``.

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
    tensor([-0.8647, -0.6321, 0., 1.])
    """
    if inplace:
        return _in_place(x, "celu_", lambda t: celu(t, alpha))
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
    tensor([-1., 0., 0., 0., 1.])
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
    tensor([-1.036, -0.03788, 0., 0.03788, 1.036])
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
    >>> import lucid.nn.functional as F
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
    tensor([0.7071])
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
        Small constant added to ``x1 - x2`` before the norm, as the
        reference does.  Default ``1e-6``.
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

        d_p(x_1, x_2) = \| x_1 - x_2 + \epsilon \|_p
        = \Big(\sum_i |x_{1,i} - x_{2,i} + \epsilon|^p\Big)^{1/p}

    The ``eps`` shift keeps coincident points off the norm's kink.  With
    ``eps=0`` they sit on it, and their gradient is 0 — the subgradient
    the reference picks — not ``nan``.  The shift used to be added to
    ``|x1 - x2|`` instead, which differs by up to ``2 * eps`` per entry
    where ``x1 < x2``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pairwise_distance
    >>> a = lucid.tensor([[1.0, 2.0]])
    >>> b = lucid.tensor([[4.0, 6.0]])
    >>> pairwise_distance(a, b, p=2.0)
    tensor([5.])
    """
    # The shared vector norm, so a zero distance (``eps=0`` at coincident
    # points) has the 0 subgradient rather than a nan one, half inputs
    # reduce in float32, and ``p=0`` counts instead of dividing by zero.
    return _l.linalg.vector_norm(x1 - x2 + eps, p, -1, keepdim)


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
    tensor([-0.5, 0., 0., 0., 0.5])
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
        If ``True``, write the result into ``x`` and return ``x``.

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
    tensor([-1., -0.5, 0., 0.5, 1.])
    """
    if inplace:
        return _in_place(x, "hardtanh_", lambda t: hardtanh(t, min_val, max_val))
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
    tensor([-10., -0.6931, -4.54e-05])
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
    tensor([-0.6667, -0.5, 0., 0.5, 0.6667])
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
        If ``True``, write the result into ``x`` and return ``x``.

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
    tensor([-99., -99., -99., 1.])
    """
    if inplace:
        return _in_place(x, "threshold_", lambda t: _threshold(t, threshold, value))
    return _threshold(x, threshold, value)


def _threshold(x: Tensor, threshold: float, value: float) -> Tensor:
    # ``x <= t ? v : x``, not ``x > t ? x : v``.  The two are the same
    # function everywhere except at NaN, where *both* comparisons are
    # false — so the first keeps the NaN and the second replaces it with
    # ``value``.  Written the second way, a NaN entering a network became
    # a 0 at the first threshold and the loss went finite with nothing to
    # show for it.  The reference is the first.  (The public function's
    # ``threshold`` argument shadows its own name, hence this helper.)
    replaced = x <= threshold
    replacement = _l.full_like(x, float(value))
    return _l.where(replaced, replacement, x)


def straight_through(hard: Tensor, soft: Tensor) -> Tensor:
    r"""Straight-through estimator — ``hard`` forward, ``soft`` backward.

    Returns a tensor numerically equal to ``hard`` whose gradient is the
    gradient of ``soft``:

    .. math::

        \mathrm{ST}(h, s) = s + \mathrm{sg}\!\big[h - s\big],

    where :math:`\mathrm{sg}` is the stop-gradient operator.  The forward
    value is exactly :math:`h`, and :math:`\partial \mathrm{ST} /
    \partial s` is the identity, so a discrete or otherwise
    non-differentiable ``hard`` can sit in the middle of a network while
    the surrounding layers still train.

    Parameters
    ----------
    hard : Tensor
        The value the forward pass should carry — typically the output of
        a non-differentiable step such as a one-hot ``argmax`` or a
        nearest-codebook lookup.
    soft : Tensor
        The differentiable surrogate whose gradient is borrowed.  Must be
        broadcast-compatible with ``hard``; in practice the two have the
        same shape.

    Returns
    -------
    Tensor
        Equal to ``hard`` in value, differentiable with respect to
        whatever ``soft`` depends on.

    Notes
    -----
    Introduced by Bengio, Léonard, and Courville, *"Estimating or
    Propagating Gradients Through Stochastic Neurons for Conditional
    Computation"* (arXiv:1308.3432).  It is the mechanism behind
    :func:`gumbel_softmax` with ``hard=True`` and behind
    :class:`lucid.nn.VectorQuantizer`, which is why it lives here rather
    than being rewritten at each call site.

    The estimator is biased — the returned gradient is not the gradient of
    the discrete function, which has none — but it is low-variance and
    empirically trains where the true gradient does not exist.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> soft = lucid.tensor([0.3, 0.7], requires_grad=True)
    >>> hard = lucid.tensor([0.0, 1.0])
    >>> out = F.straight_through(hard, soft)
    >>> out.tolist()               # the forward value is ``hard``
    [0.0, 1.0]
    >>> out.sum().backward()
    >>> soft.grad.tolist()         # gradient passed through untouched
    [1.0, 1.0]
    """
    return soft + (hard - soft).detach()


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

    # Build a one-hot at argmax, then re-attach ``y_soft``'s gradient.
    idx: Tensor = y_soft.argmax(dim=dim, keepdim=True)
    y_hard: Tensor = _l.zeros_like(y_soft)
    y_hard = y_hard.scatter(dim, idx, _l.ones_like(idx, dtype=y_soft.dtype))
    return straight_through(y_hard, y_soft)


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
        If ``True``, ``x`` itself receives the result and is returned.

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
    tensor([-0.2292, 0., 1.])
    """
    if inplace:
        return _in_place(x, "rrelu_", lambda t: rrelu(t, lower, upper, training))
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


# ── In-place variants ─────────────────────────────────────────────────────────
# Each is its functional with ``inplace`` passed on (default ``True``), so
# ``F.elu_(x)`` and ``F.elu(x, inplace=True)`` are one call: ``_in_place``.


def relu_(x: Tensor, inplace: bool = True) -> Tensor:
    """In-place ReLU — ``x`` is overwritten with ``max(0, x)``.

    Backed by the engine's native in-place ReLU kernel when no autograd
    graph has to be extended (no temporary buffer).

    Parameters
    ----------
    x : Tensor
        Input; modified in place and also returned.
    inplace : bool, optional
        ``False`` returns :func:`relu`'s fresh result instead.  Default ``True``.

    Returns
    -------
    Tensor
        The same ``x`` tensor, now holding ``max(0, x)``.
    """
    return relu(x, inplace)


def elu_(x: Tensor, alpha: float = 1.0, inplace: bool = True) -> Tensor:
    r"""In-place ELU — overwrites ``x`` with :math:`\text{ELU}(x, \alpha)`.

    Parameters
    ----------
    x : Tensor
        Input; modified in place.
    alpha : float, optional
        Negative-side scale parameter :math:`\alpha`.  Default ``1.0``.
    inplace : bool, optional
        ``False`` returns :func:`elu`'s fresh result instead.  Default ``True``.

    Returns
    -------
    Tensor
        The same ``x`` tensor, now holding ``elu(x, alpha)``.
    """
    return elu(x, alpha, inplace)


def selu_(x: Tensor, inplace: bool = True) -> Tensor:
    """In-place SELU — overwrites ``x`` with ``selu(x)``.

    Uses the canonical SELU coefficients (Klambauer 2017); intended
    for use only inside Self-Normalising Networks where the input was
    initialised with LeCun-normal and downstream blocks expect the
    SELU fixed-point statistics.

    Parameters
    ----------
    x : Tensor
        Input; modified in place.
    inplace : bool, optional
        ``False`` returns :func:`selu`'s fresh result instead.  Default ``True``.

    Returns
    -------
    Tensor
        The same ``x`` tensor, now holding ``selu(x)``.
    """
    return selu(x, inplace)


def leaky_relu_(
    x: Tensor, negative_slope: float = 0.01, inplace: bool = True
) -> Tensor:
    r"""In-place Leaky ReLU — overwrites ``x`` with the leaky-rectified result.

    .. math::

        y_i = \begin{cases} x_i & x_i \ge 0 \\ \text{negative\_slope} \cdot x_i & x_i < 0 \end{cases}

    Parameters
    ----------
    x : Tensor
        Input; modified in place.
    negative_slope : float, optional
        Slope applied to the negative side.  Default ``0.01``.
    inplace : bool, optional
        ``False`` returns :func:`leaky_relu`'s fresh result instead.
        Default ``True``.

    Returns
    -------
    Tensor
        The same ``x`` tensor, now holding ``leaky_relu(x, negative_slope)``.
    """
    return leaky_relu(x, negative_slope, inplace)


def hardtanh_(
    x: Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = True,
) -> Tensor:
    """In-place hardtanh — clamps ``x`` to ``[min_val, max_val]`` in place.

    Parameters
    ----------
    x : Tensor
        Input; modified in place.
    min_val : float, optional
        Lower clamp bound.  Default ``-1.0``.
    max_val : float, optional
        Upper clamp bound.  Default ``1.0``.
    inplace : bool, optional
        ``False`` returns :func:`hardtanh`'s fresh result instead.
        Default ``True``.

    Returns
    -------
    Tensor
        The same ``x`` tensor, now clamped.
    """
    return hardtanh(x, min_val, max_val, inplace)


def threshold_(
    x: Tensor,
    threshold_val: float,
    value: float,
    inplace: bool = True,
) -> Tensor:
    """In-place threshold — replace entries ``<= threshold_val`` with ``value``.

    Parameters
    ----------
    x : Tensor
        Input; modified in place.
    threshold_val : float
        Cutoff; positions where ``x <= threshold_val`` are reset.
    value : float
        Replacement value for clipped positions.
    inplace : bool, optional
        ``False`` returns :func:`threshold`'s fresh result instead.
        Default ``True``.

    Returns
    -------
    Tensor
        The same ``x`` tensor, with sub-threshold positions set to
        ``value`` and the rest unchanged.
    """
    return threshold(x, threshold_val, value, inplace)


def rrelu_(
    x: Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
    inplace: bool = True,
) -> Tensor:
    """In-place RReLU — randomised leaky ReLU written back into ``x``.

    During training the negative-slope is sampled per-element from a
    uniform :math:`U(\\text{lower}, \\text{upper})`; at inference the
    midpoint :math:`(\\text{lower} + \\text{upper}) / 2` is used.

    Parameters
    ----------
    x : Tensor
        Input; modified in place.
    lower : float, optional
        Lower bound of the uniform slope sampler.  Default ``1/8``.
    upper : float, optional
        Upper bound of the uniform slope sampler.  Default ``1/3``.
    training : bool, optional
        When ``False`` (default) uses the deterministic midpoint slope;
        when ``True`` samples a fresh slope per element.
    inplace : bool, optional
        ``False`` returns :func:`rrelu`'s fresh result instead.
        Default ``True``.

    Returns
    -------
    Tensor
        The same ``x`` tensor, now holding ``rrelu(x, lower, upper, training)``.
    """
    return rrelu(x, lower, upper, training, inplace)


def symlog(x: Tensor) -> Tensor:
    r"""Symmetric logarithm — compress magnitude, keep sign and zero.

    .. math::

        \mathrm{symlog}(x) = \mathrm{sign}(x)\,\ln(|x| + 1)

    Parameters
    ----------
    x : Tensor
        Any shape.

    Returns
    -------
    Tensor
        Same shape.

    Notes
    -----
    Behaves like the identity near zero and like a logarithm far from it,
    so a single network can be trained on targets whose scale is not known
    in advance — rewards that are ``0.01`` in one environment and
    ``10000`` in another land in the same range without per-task
    normalisation.  That is what it is for in DreamerV3.

    The inverse is :func:`symexp`, and the pair is exact: ``symexp`` of
    ``symlog`` returns the input.  Implemented through ``log1p`` rather
    than ``log(1 + ·)`` so small inputs keep their precision.

    See Also
    --------
    symexp : The inverse transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> F.symlog(lucid.tensor([-100.0, 0.0, 100.0]))
    tensor([-4.615, 0., 4.615])
    """
    return _l.sign(x) * _l.log1p(_l.abs(x))


def symexp(x: Tensor) -> Tensor:
    r"""Inverse of :func:`symlog`.

    .. math::

        \mathrm{symexp}(x) = \mathrm{sign}(x)\big(\exp(|x|) - 1\big)

    Parameters
    ----------
    x : Tensor
        Any shape.

    Returns
    -------
    Tensor
        Same shape.

    Notes
    -----
    Used to read a prediction back out into the original units — a value
    head trained against ``symlog`` targets predicts in the compressed
    space, and this is what turns that back into a return.

    Uses ``expm1`` rather than ``exp(·) - 1``, which loses precision for
    small arguments exactly where ``symlog`` sends most of its inputs.

    See Also
    --------
    symlog : The forward transform.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> round(float(F.symexp(F.symlog(lucid.tensor([1234.0]))).item()), 2)
    1234.0
    """
    return _l.sign(x) * _l.expm1(_l.abs(x))
