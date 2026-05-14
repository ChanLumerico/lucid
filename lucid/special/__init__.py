"""``lucid.special`` — special mathematical functions.

This sub-package adds functions that complement the elementary math
already available at the top level (``erf`` / ``sinc`` / ``lgamma`` /
``digamma`` / ``i0`` / ``log1p`` / ``logit`` / ``xlogy`` / ``erfc`` /
``erfinv`` / ``expm1`` …).  Per **H8** there is exactly one canonical
path per op — those stay accessible only at the top level, and the
extra entries below stay accessible only via ``lucid.special.<name>``.

Implemented (all pure-Python composites over engine ops — no C++ work):

* ``erfcx``                 — scaled complementary error: ``exp(x²) · erfc(x)``
* ``i0e``                   — exponentially-scaled modified Bessel I₀:
                              ``exp(-|x|) · i0(x)``
* ``i1``                    — modified Bessel of order 1 (Abramowitz polynomial)
* ``i1e``                   — exponentially-scaled ``i1``
* ``ndtr``                  — normal CDF ``Φ(x) = ½·(1 + erf(x/√2))``
* ``ndtri``                 — inverse normal CDF (Beasley-Springer-Moro)
* ``log_ndtr``              — numerically-stable ``log(Φ(x))``
* ``xlog1py``               — ``x · log1p(y)`` with ``0·log1p(0) = 0``
* ``entr``                  — entropy element ``-x · log(x)``, 0 at x=0
* ``multigammaln``          — log of the multivariate gamma function
* ``polygamma``             — derivatives of ``digamma``; n=0 falls through
                              to ``lucid.digamma``, n=1 (trigamma) uses the
                              standard series; n ≥ 2 raises ``NotImplementedError``
* ``spherical_bessel_j0``   — ``sin(x)/x`` with continuous extension at 0

Deferred (would need oscillatory / Hurwitz-zeta approximations or
arbitrary-order Bessel machinery — enable in a follow-up if anyone
actually asks): ``iv``, ``ive``, ``bessel_j0``/``j1``/``y0``/``y1``,
``modified_bessel_k0``/``k1`` (and scaled variants), ``zeta``, the
orthogonal-polynomial family (Chebyshev / Legendre / Hermite /
Laguerre / Jacobi).
"""

import math
from typing import TYPE_CHECKING, Callable

import lucid

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Error-function family ──────────────────────────────────────────────────


def erfcx(x: Tensor) -> Tensor:
    r"""Scaled complementary error function.

    Computes :math:`\mathrm{erfcx}(x) = e^{x^2}\, \mathrm{erfc}(x)`, the
    Mills-ratio-friendly scaling of the complementary error function.
    Whereas :math:`\mathrm{erfc}(x)` underflows to zero for moderately
    large :math:`x`, ``erfcx`` decays only as :math:`1/(x\sqrt\pi)` and
    remains finite, which is essential for tail-probability evaluations
    of Gaussian-related distributions.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`e^{x^2}\,\mathrm{erfc}(x)` element-wise, same shape and
        dtype as ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \mathrm{erfcx}(x) = e^{x^2}\,\mathrm{erfc}(x)
                         = e^{x^2} \cdot \frac{2}{\sqrt{\pi}}
                           \int_x^\infty e^{-t^2}\, dt.

    The implementation forms the product of ``exp(x*x)`` and ``erfc(x)``
    directly, which is accurate for :math:`|x| \lesssim 5`.  For very
    large positive ``x`` the two factors hit opposite floating-point
    extremes and accuracy degrades — use a dedicated continued-fraction
    expansion if extreme-value precision is required.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import erfcx
    >>> erfcx(lucid.tensor([0.0, 1.0, 5.0]))
    Tensor([1.0000, 0.4276, 0.1107])
    """
    return lucid.exp(x * x) * lucid.erfc(x)


# ── Modified Bessel I₀ / I₁ family ─────────────────────────────────────────


def i0e(x: Tensor) -> Tensor:
    r"""Exponentially scaled modified Bessel function of order 0.

    Computes :math:`I_0(x)\, e^{-|x|}`, the standard exponentially-scaled
    variant of the modified Bessel function of the first kind.  ``I_0``
    itself grows as :math:`e^{|x|}/\sqrt{|x|}` and therefore overflows
    floating-point representation for moderately large arguments; the
    scaled form is bounded by 1 for :math:`x \ge 0` and is the natural
    quantity to manipulate in log-density computations (von Mises /
    Bessel distributions, Rice distribution, etc.).

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_0(x)\, e^{-|x|}` element-wise, same shape and dtype as
        ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        I_0(x) e^{-|x|}, \quad
        I_0(x) = \sum_{k=0}^\infty \frac{(x/2)^{2k}}{(k!)^2}.

    The function is even and equals 1 at ``x = 0``.  For
    :math:`x \to \infty`,
    :math:`I_0(x) e^{-x} \sim 1/\sqrt{2\pi x}`.  Related identities:
    ``i0e(0) = 1`` and ``i0e(x) <= 1`` for all real ``x``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import i0e
    >>> i0e(lucid.tensor([0.0, 1.0, 5.0, 20.0]))
    Tensor([1.0000, 0.4658, 0.1835, 0.0897])
    """
    return lucid.exp(-lucid.abs(x)) * lucid.i0(x)


# Abramowitz & Stegun §9.8 — same polynomial form as ``lucid.i0`` but for I₁.
_I1_SMALL_COEFFS: list[float] = [
    0.5,
    0.87890594,
    0.51498869,
    0.15084934,
    0.02658733,
    0.00301532,
    0.00032411,
]
_I1_LARGE_COEFFS: list[float] = [
    0.39894228,
    -0.03988024,
    -0.00362018,
    0.00163801,
    -0.01031555,
    0.02282967,
    -0.02895312,
    0.01787654,
    -0.00420059,
]


def i1(x: Tensor) -> Tensor:
    r"""Modified Bessel function of the first kind, order 1.

    Computes :math:`I_1(x)`, the order-one solution of the modified
    Bessel equation :math:`x^2 y'' + x y' - (x^2 + 1) y = 0` that is
    regular at the origin.  Used in Rician statistics, Bessel /
    von Mises – Fisher distributions, and rotational-symmetric kernels.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_1(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Series definition:

    .. math::

        I_1(x) = \sum_{k=0}^\infty
                 \frac{1}{k!\,(k+1)!}\left(\frac{x}{2}\right)^{2k+1}.

    The implementation uses the two-branch Abramowitz & Stegun §9.8
    polynomial approximation: a power series in :math:`(x/3.75)^2` for
    :math:`|x| \le 3.75`, and an asymptotic expansion in
    :math:`3.75/|x|` scaled by :math:`e^{|x|}/\sqrt{|x|}` for
    :math:`|x| > 3.75`.  Accuracy is roughly :math:`10^{-7}` over the
    real line.  ``I_1`` is odd, ``I_1(0) = 0``, and
    :math:`I_1(x) \sim e^{x}/\sqrt{2\pi x}` as :math:`x \to \infty`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import i1
    >>> i1(lucid.tensor([0.0, 1.0, 5.0]))
    Tensor([0.0000, 0.5652, 24.3356])
    """
    ax = lucid.abs(x)
    ax_safe = lucid.where(ax == lucid.zeros_like(ax), lucid.full_like(ax, 1.0), ax)

    # Small-argument branch: x · poly((x / 3.75)²)
    t_small = (x * (1.0 / 3.75)) ** 2
    val_small = lucid.full_like(x, _I1_SMALL_COEFFS[0])
    t_pow = lucid.ones_like(x)
    for c in _I1_SMALL_COEFFS[1:]:
        t_pow = t_pow * t_small
        val_small = val_small + c * t_pow
    val_small = val_small * x  # I₁ is odd

    # Large-argument branch: sign(x) · exp(|x|)/√|x| · poly(3.75/|x|)
    y_large = 3.75 / ax_safe
    val_large = lucid.full_like(x, _I1_LARGE_COEFFS[0])
    y_pow = lucid.ones_like(x)
    for c in _I1_LARGE_COEFFS[1:]:
        y_pow = y_pow * y_large
        val_large = val_large + c * y_pow
    val_large = val_large * lucid.exp(ax_safe) / lucid.sqrt(ax_safe)
    val_large = val_large * lucid.sign(x)

    return lucid.where(ax <= 3.75, val_small, val_large)


def i1e(x: Tensor) -> Tensor:
    r"""Exponentially scaled modified Bessel function of order 1.

    Computes :math:`I_1(x)\, e^{-|x|}`, the bounded counterpart of
    :func:`i1`.  Because :math:`I_1` itself grows exponentially with
    :math:`|x|`, this scaled form is the safer building block for
    log-likelihood expressions (Rice / Skellam distributions and
    similar) where evaluating ``I_1`` directly would overflow.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_1(x)\, e^{-|x|}` element-wise, same shape and dtype as
        ``x``.

    Notes
    -----
    Definition:

    .. math::

        I_1(x)\, e^{-|x|}.

    The function is odd in ``x``, equals 0 at ``x = 0``, and approaches
    :math:`1/\sqrt{2\pi x}` as :math:`x \to \infty`.  Together with
    :func:`i0e` this forms the standard scaled-Bessel pair for stable
    evaluation of rotational-symmetric likelihoods.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import i1e
    >>> i1e(lucid.tensor([0.0, 1.0, 5.0]))
    Tensor([0.0000, 0.2079, 0.1640])
    """
    return lucid.exp(-lucid.abs(x)) * i1(x)


# ── Normal-distribution helpers ────────────────────────────────────────────

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def ndtr(x: Tensor) -> Tensor:
    r"""Standard normal cumulative distribution function.

    Evaluates :math:`\Phi(x) = P(Z \le x)` where :math:`Z` is a standard
    normal random variable.  Implemented as a thin wrapper around the
    error function via the textbook identity
    :math:`\Phi(x) = \tfrac{1}{2}\,(1 + \mathrm{erf}(x/\sqrt{2}))`.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\Phi(x)` element-wise, same shape and dtype as ``x``;
        values lie in ``(0, 1)``.

    Notes
    -----
    Definition:

    .. math::

        \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-t^2/2}\, dt
                = \frac{1}{2}\left(1 + \mathrm{erf}\!\left(
                  \frac{x}{\sqrt{2}}\right)\right).

    For ``x`` deep in the left tail, ``ndtr`` underflows to ``0`` and
    losses precision; prefer :func:`log_ndtr` for log-domain work.
    Special values: :math:`\Phi(-\infty) = 0`, :math:`\Phi(0) = 0.5`,
    :math:`\Phi(+\infty) = 1`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import ndtr
    >>> ndtr(lucid.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]))
    Tensor([0.0013, 0.1587, 0.5000, 0.8413, 0.9987])
    """
    return 0.5 * (1.0 + lucid.erf(x * _INV_SQRT2))


def log_ndtr(x: Tensor) -> Tensor:
    r"""Numerically stable logarithm of the standard normal CDF.

    Computes :math:`\log \Phi(x)` without losing precision in the deep
    left tail, where the direct composition ``log(ndtr(x))`` underflows.
    Essential for log-likelihood evaluation of probit / Tobit / censored
    models that need to handle :math:`x \ll 0` cleanly.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\log \Phi(x)` element-wise, same shape and dtype as
        ``x``; values lie in :math:`(-\infty, 0]`.

    Notes
    -----
    The implementation switches strategies on a branch boundary at
    ``x = -1``:

    .. math::

        \log \Phi(x) = \begin{cases}
            \log \Phi(x), & x \ge -1 \\[2pt]
            \log\!\left(\tfrac{1}{2}\,
                  \mathrm{erfc}(-x/\sqrt{2})\right), & x < -1.
        \end{cases}

    For :math:`x \ge -1` we have :math:`\Phi(x) \gtrsim 0.16` so the
    direct ``log(ndtr(x))`` is safe.  Below :math:`x = -1` the identity
    :math:`\Phi(x) = \tfrac{1}{2}\,\mathrm{erfc}(-x/\sqrt{2})` is used,
    sidestepping the catastrophic cancellation in
    :math:`1 + \mathrm{erf}(x/\sqrt{2})`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import log_ndtr
    >>> log_ndtr(lucid.tensor([-10.0, -1.0, 0.0, 1.0]))
    Tensor([-52.6651, -1.8410, -0.6931, -0.1727])
    """
    direct = lucid.log(ndtr(x))
    asymp = lucid.log(0.5 * lucid.erfc(-x * _INV_SQRT2))
    return lucid.where(x >= lucid.full_like(x, -1.0), direct, asymp)


# Beasley-Springer-Moro coefficients for the inverse normal CDF.  Hosted as
# module-level constants so ``ndtri`` stays a tight composite.
_NDTRI_A = (
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.383577518672690e2,
    -3.066479806614716e1,
    2.506628277459239,
)
_NDTRI_B = (
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
)
_NDTRI_C = (
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
)
_NDTRI_D = (
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
)


def ndtri(p: Tensor) -> Tensor:
    r"""Inverse standard normal CDF (probit / quantile function).

    Computes :math:`\Phi^{-1}(p)`, the quantile function of the standard
    normal distribution.  This is the workhorse for sampling Gaussian
    variates from uniforms via inverse-CDF, for probit regression, and
    for converting tail probabilities back to standard scores.

    Parameters
    ----------
    p : Tensor
        Input tensor of probabilities in :math:`(0, 1)`; any
        floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\Phi^{-1}(p)` element-wise, same shape and dtype as
        ``p``; values are unbounded reals.

    Notes
    -----
    The implementation uses the Beasley-Springer-Moro rational
    approximation.  The unit interval is split into a central region
    :math:`[p_{lo}, p_{hi}]` with :math:`p_{lo} = 0.02425`,
    :math:`p_{hi} = 1 - p_{lo}`, in which a rational polynomial in
    :math:`q = p - 0.5` is evaluated, and two symmetric tail regions in
    which a rational polynomial in
    :math:`r = \sqrt{-2 \log\min(p, 1-p)}` is used:

    .. math::

        \Phi^{-1}(p) \approx \begin{cases}
            q \cdot \frac{N_c(q^2)}{D_c(q^2)},
                & p \in [p_{lo}, p_{hi}] \\[4pt]
            \pm \frac{N_t(r)}{D_t(r)},
                & \text{otherwise}.
        \end{cases}

    The approximation is accurate to roughly :math:`1.15 \times 10^{-9}`
    over its supported domain.  Boundary behaviour:
    :math:`\Phi^{-1}(0.5) = 0`, and the function blows up to
    :math:`\mp \infty` at the endpoints.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import ndtri
    >>> ndtri(lucid.tensor([0.025, 0.5, 0.975]))
    Tensor([-1.9600, 0.0000, 1.9600])
    """
    plow = 0.02425
    phigh = 1.0 - plow

    # Central region.
    q = p - 0.5
    r_central = q * q
    num_c = (
        (
            ((_NDTRI_A[0] * r_central + _NDTRI_A[1]) * r_central + _NDTRI_A[2])
            * r_central
            + _NDTRI_A[3]
        )
        * r_central
        + _NDTRI_A[4]
    ) * r_central + _NDTRI_A[5]
    den_c = (
        (
            ((_NDTRI_B[0] * r_central + _NDTRI_B[1]) * r_central + _NDTRI_B[2])
            * r_central
            + _NDTRI_B[3]
        )
        * r_central
        + _NDTRI_B[4]
    ) * r_central + 1.0
    central = q * num_c / den_c

    # Tails: same polynomial form on r = √(-2·log(p_tail)), with a sign flip
    # at the upper tail.  Compute the lower-tail value for every element;
    # in the upper-tail branch we pre-flip ``p`` to ``1 - p`` so the same
    # polynomial applies, then negate the answer.
    p_low_tail = p
    p_high_tail = 1.0 - p
    p_tail_in = lucid.where(p < lucid.full_like(p, 0.5), p_low_tail, p_high_tail)
    # Clamp to keep log finite even when callers pass ``p == 0`` / ``1``.
    p_tail_safe = lucid.clip(p_tail_in, 1e-300, 1.0)
    r_tail = lucid.sqrt(-2.0 * lucid.log(p_tail_safe))
    num_t = (
        (
            ((_NDTRI_C[0] * r_tail + _NDTRI_C[1]) * r_tail + _NDTRI_C[2]) * r_tail
            + _NDTRI_C[3]
        )
        * r_tail
        + _NDTRI_C[4]
    ) * r_tail + _NDTRI_C[5]
    den_t = (
        ((_NDTRI_D[0] * r_tail + _NDTRI_D[1]) * r_tail + _NDTRI_D[2]) * r_tail
        + _NDTRI_D[3]
    ) * r_tail + 1.0
    tail_low = num_t / den_t
    tail = lucid.where(p < lucid.full_like(p, 0.5), tail_low, -tail_low)

    in_central = (p > lucid.full_like(p, plow)) & (p < lucid.full_like(p, phigh))
    return lucid.where(in_central, central, tail)


# ── x-times-log-of-something helpers ───────────────────────────────────────


def xlog1py(x: Tensor, y: Tensor) -> Tensor:
    r"""Safe product :math:`x \log(1 + y)` with limit-convention zero handling.

    Computes :math:`x \log(1 + y)` element-wise but enforces the
    convention :math:`0 \cdot \log(1 + 0) = 0`, and more generally
    propagates the zero whenever :math:`x = 0` regardless of ``y``.
    Mirrors :func:`lucid.xlogy` but uses :math:`\log(1 + y)` instead of
    :math:`\log y`, which is the right primitive for log-densities of
    distributions expressed in terms of small offsets (Negative Binomial
    log-likelihood, Beta survival functions, etc.).

    Parameters
    ----------
    x : Tensor
        Multiplier tensor.
    y : Tensor
        Argument of ``log1p``; broadcast-compatible with ``x``.

    Returns
    -------
    Tensor
        :math:`x \log(1 + y)` element-wise, broadcast to the common
        shape of ``x`` and ``y``.

    Notes
    -----
    Mathematical definition with the limit convention:

    .. math::

        \text{xlog1py}(x, y) = \begin{cases}
            0, & x = 0 \\[2pt]
            x \log(1 + y), & \text{otherwise}.
        \end{cases}

    Using :math:`\log(1 + y)` avoids precision loss for small ``y``
    where ``1 + y`` would round to ``1.0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import xlog1py
    >>> x = lucid.tensor([0.0, 1.0, 2.0])
    >>> y = lucid.tensor([0.0, 1.0, 3.0])
    >>> xlog1py(x, y)
    Tensor([0.0000, 0.6931, 2.7726])
    """
    safe_y = lucid.where(y == lucid.zeros_like(y), lucid.full_like(y, 0.0), y)
    out = x * lucid.log1p(safe_y)
    return lucid.where(x == lucid.zeros_like(x), lucid.full_like(out, 0.0), out)


def entr(x: Tensor) -> Tensor:
    r"""Element-wise entropy kernel :math:`-x \log x`.

    Returns the per-element contribution to Shannon entropy.  This is
    the standard building block for evaluating :math:`H(p) =
    \sum_i \mathrm{entr}(p_i)` for a discrete distribution, with the
    limit convention :math:`0 \log 0 = 0` applied at the boundary.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.  Values outside
        :math:`[0, \infty)` produce ``NaN``.

    Returns
    -------
    Tensor
        Element-wise :math:`-x \log x` with the limit convention at
        ``x = 0``; same shape and dtype as ``x``.

    Notes
    -----
    Definition:

    .. math::

        \mathrm{entr}(x) = \begin{cases}
            -x \log x, & x > 0 \\[2pt]
            0,         & x = 0 \\[2pt]
            \mathrm{NaN}, & x < 0.
        \end{cases}

    The function is concave with maximum :math:`1/e \approx 0.3679` at
    :math:`x = 1/e`.  It vanishes at both ``x = 0`` (limit) and
    ``x = 1``.  Related quantities include :func:`rel_entr` (point-wise
    relative entropy) and :func:`kl_div` (KL kernel).

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import entr
    >>> entr(lucid.tensor([0.0, 0.5, 1.0, 2.0]))
    Tensor([0.0000, 0.3466, 0.0000, -1.3863])
    """
    safe_x = lucid.where(x > lucid.zeros_like(x), x, lucid.full_like(x, 1.0))
    val = -safe_x * lucid.log(safe_x)
    val = lucid.where(x == lucid.zeros_like(x), lucid.full_like(val, 0.0), val)
    val = lucid.where(x < lucid.zeros_like(x), lucid.full_like(val, float("nan")), val)
    return val


# ── Gamma family ───────────────────────────────────────────────────────────


def multigammaln(a: Tensor, p: int) -> Tensor:
    r"""Logarithm of the multivariate gamma function :math:`\Gamma_p(a)`.

    Computes :math:`\log \Gamma_p(a)`, the natural generalisation of
    ``lgamma`` that appears in the normalising constants of the Wishart
    and inverse-Wishart distributions over positive-definite matrices,
    and in matrix-Bayes log-evidence formulas.

    Parameters
    ----------
    a : Tensor
        Real argument; any floating-point dtype.  Each entry must
        satisfy :math:`a > (p - 1)/2` for the function to be finite
        (otherwise the underlying ``lgamma`` returns ``inf``).
    p : int
        Dimensionality of the matrix argument; must be a positive
        integer.

    Returns
    -------
    Tensor
        :math:`\log \Gamma_p(a)` element-wise over ``a``, same shape
        and dtype as ``a``.

    Notes
    -----
    Definition:

    .. math::

        \log \Gamma_p(a)
            = \frac{p(p-1)}{4}\,\log \pi
              + \sum_{i=1}^{p} \log \Gamma\!\left(a + \frac{1 - i}{2}
              \right).

    For ``p = 1`` this reduces to :math:`\log \Gamma(a)`.  Raises
    ``ValueError`` if ``p < 1``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import multigammaln
    >>> multigammaln(lucid.tensor([3.0, 5.0]), p=2)
    Tensor([1.7918, 5.4538])
    """
    if int(p) < 1:
        raise ValueError(f"multigammaln requires p >= 1, got {p}")
    accum = lucid.zeros_like(a)
    for i in range(1, int(p) + 1):
        accum = accum + lucid.lgamma(a + (1.0 - float(i)) / 2.0)
    return float(p * (p - 1)) / 4.0 * math.log(math.pi) + accum


def polygamma(n: int, x: Tensor) -> Tensor:
    r"""Polygamma function :math:`\psi^{(n)}(x)`.

    The ``n``-th derivative of the digamma function.  Polygamma
    functions appear in the derivatives of log-partition functions of
    the Gamma / Beta / Dirichlet families and are needed for evaluating
    Fisher information matrices for these distributions analytically.

    Parameters
    ----------
    n : int
        Non-negative integer order.  Only :math:`n \in \{0, 1, 2, 3\}`
        are supported; ``n = 0`` recovers :func:`lucid.digamma`,
        ``n = 1`` is the trigamma function.  Higher orders raise
        ``NotImplementedError``.
    x : Tensor
        Real argument; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\psi^{(n)}(x)` element-wise, same shape and dtype as
        ``x``.

    Notes
    -----
    Definition (for ``n >= 1``):

    .. math::

        \psi^{(n)}(x) = \frac{d^{n+1}}{dx^{n+1}} \log \Gamma(x)
                     = (-1)^{n+1}\, n!\, \sum_{k=0}^\infty
                       \frac{1}{(x + k)^{n+1}}.

    Implementation: shift ``x`` upward by :math:`K = 6` using the
    recurrence

    .. math::

        \psi^{(n)}(x) = \psi^{(n)}(x + 1)
                      + \frac{(-1)^{n+1} n!}{x^{n+1}},

    accumulate the per-step corrections, then evaluate the Bernoulli
    asymptotic series at the shifted argument where it is
    well-conditioned.  Accuracy is roughly seven decimal digits across
    the positive-real regime.

    Raises ``ValueError`` for ``n < 0`` and ``NotImplementedError`` for
    ``n >= 4``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import polygamma
    >>> polygamma(1, lucid.tensor([1.0, 2.0, 5.0]))
    Tensor([1.6449, 0.6449, 0.2213])
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"polygamma: n must be ≥ 0, got {n}")
    if n == 0:
        return lucid.digamma(x)
    if n > 3:
        raise NotImplementedError(
            f"polygamma: only n ∈ {{0, 1, 2, 3}} are wired; got n={n}. "
            "Higher orders need an extended Bernoulli-series term table."
        )

    # ── recurrence shift: K=6 steps so the asymptotic series at x+6 is
    #    accurate to ~7 digits across the typical positive-real regime. ──
    K: int = 6
    sign: float = 1.0 if n % 2 == 1 else -1.0  # (−1)^(n+1) — n odd → +.
    # n!  pre-computed as a Python int; ints up to 3! = 6 fit happily in float.
    n_fact: float = float(math.factorial(n))
    correction: Tensor = lucid.zeros_like(x)
    for k in range(K):
        correction = correction + sign * n_fact / ((x + float(k)) ** (n + 1))
    xr: Tensor = x + float(K)

    # ── asymptotic series at xr.  For n ≥ 1 (Wikipedia, "Polygamma function"):
    #   ψ⁽ⁿ⁾(x) ≈ (−1)^(n+1) · [ (n−1)!/x^n + n!/(2·x^(n+1))
    #                          + ∑_{k≥1} B_{2k} · (2k+n−1)!/((2k)!) ·
    #                                     1/x^(2k+n) ]
    # Powers in the series: m, m+1, m+2, m+4, m+6 (for k=1..3).  The
    # earlier hand-tuned trigamma branch used the same five-term truncation;
    # we generalise with closed-form per-n coefficients. ──
    r: Tensor = 1.0 / xr
    inv_xn: Tensor = r**n  # 1 / xr^n.
    inv_xnp1: Tensor = inv_xn * r  # 1 / xr^(n+1).
    inv_xnp2: Tensor = inv_xnp1 * r  # 1 / xr^(n+2)  ← k=1 term.
    inv_xnp4: Tensor = inv_xnp2 * r * r  # 1 / xr^(n+4)  ← k=2 term.
    inv_xnp6: Tensor = inv_xnp4 * r * r  # 1 / xr^(n+6)  ← k=3 term.

    # Coefficients per n.  The leading two are (n−1)! and n!/2; the
    # Bernoulli triplet is B_{2k}·(2k+n−1)!/(2k)! for k=1..3.
    if n == 1:
        c_lead: float = 1.0  # 0! = 1.
        c_half: float = 0.5  # 1!/2 = 1/2.
        # k=1: B₂·2!/2! = 1/6, k=2: B₄·4!/4! = −1/30, k=3: B₆·6!/6! = 1/42.
        c1: float = 1.0 / 6.0
        c2: float = -1.0 / 30.0
        c3: float = 1.0 / 42.0
    elif n == 2:
        c_lead = 1.0  # 1! = 1.
        c_half = 1.0  # 2!/2 = 1.
        # k=1: B₂·3!/2! = (1/6)·3 = 1/2.
        # k=2: B₄·5!/4! = (−1/30)·5 = −1/6.
        # k=3: B₆·7!/6! = (1/42)·7 = 1/6.
        c1 = 0.5
        c2 = -1.0 / 6.0
        c3 = 1.0 / 6.0
    else:  # n == 3
        c_lead = 2.0  # 2! = 2.
        c_half = 3.0  # 3!/2 = 3.
        # k=1: B₂·4!/2! = (1/6)·12 = 2.
        # k=2: B₄·6!/4! = (−1/30)·30 = −1.
        # k=3: B₆·8!/6! = (1/42)·56 = 4/3.
        c1 = 2.0
        c2 = -1.0
        c3 = 4.0 / 3.0

    series: Tensor = (
        c_lead * inv_xn
        + c_half * inv_xnp1
        + c1 * inv_xnp2
        + c2 * inv_xnp4
        + c3 * inv_xnp6
    )
    asym: Tensor = sign * series
    return asym + correction


# ── Spherical Bessel ───────────────────────────────────────────────────────


def spherical_bessel_j0(x: Tensor) -> Tensor:
    r"""Spherical Bessel function of the first kind, order 0.

    Computes :math:`j_0(x) = \sin x / x` with the continuous extension
    :math:`j_0(0) = 1`.  Arises in 3D wave / scattering problems and is
    the radial component of plane waves expanded in spherical harmonics.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`j_0(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        j_0(x) = \frac{\sin x}{x}, \qquad j_0(0) = 1.

    The implementation guards the removable singularity at the origin
    by substituting ``x = 1`` into the division and patching the result
    with ``where(x == 0, 1, ...)``, so the function is well-defined and
    differentiable across the origin.  Related to the normalised sinc:
    :math:`j_0(x) = \mathrm{sinc}(x / \pi)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import spherical_bessel_j0
    >>> spherical_bessel_j0(lucid.tensor([0.0, 1.0, 3.14159265]))
    Tensor([1.0000, 0.8415, 0.0000])
    """
    is_zero = x == lucid.zeros_like(x)
    safe_x = lucid.where(is_zero, lucid.full_like(x, 1.0), x)
    val = lucid.sin(safe_x) / safe_x
    return lucid.where(is_zero, lucid.full_like(x, 1.0), val)


# ── Orthogonal polynomials ─────────────────────────────────────────────────
#
# All seven families share the same recurrence shape ``P_{n+1} = α(x, n)·P_n
# + β(x, n)·P_{n-1}`` with constant coefficients per family.  We unfold the
# recurrence into a Python ``for`` loop because ``n`` is a static integer
# (the reference framework's signature requires it as a Python int).  The
# loop produces ``n+1`` engine ops in the autograd graph — fine for the
# small ``n`` values typical of orthogonal-polynomial use (≤ 32 in
# practice).


def _ortho_poly(
    x: Tensor,
    n: int,
    p0: Tensor,
    p1: Tensor,
    advance: Callable[[Tensor, Tensor, int], Tensor],
) -> Tensor:
    """Generic orthogonal-polynomial recurrence driver.

    ``advance(p_n, p_nm1, k)`` returns ``P_{k+1}`` given ``P_k`` and
    ``P_{k-1}``.  ``k`` is the index of ``p_n`` (so ``advance`` is called
    with ``k = 1, 2, ..., n-1`` to step from ``P_1`` to ``P_n``).
    """
    if n < 0:
        raise ValueError("orthogonal polynomial order must be >= 0")
    if n == 0:
        return p0
    if n == 1:
        return p1
    p_prev: Tensor = p0
    p_curr: Tensor = p1
    for k in range(1, n):
        p_next = advance(p_curr, p_prev, k)
        p_prev, p_curr = p_curr, p_next
    return p_curr


def chebyshev_polynomial_t(x: Tensor, n: int) -> Tensor:
    r"""Chebyshev polynomial of the first kind, :math:`T_n(x)`.

    The Chebyshev T polynomials are orthogonal on :math:`[-1, 1]` with
    weight :math:`1/\sqrt{1 - x^2}` and underlie spectral methods,
    minimax polynomial approximation, and Clenshaw-Curtis quadrature.

    Parameters
    ----------
    x : Tensor
        Argument tensor; any floating-point dtype.  Customarily
        evaluated on :math:`[-1, 1]` but defined for all real ``x``.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`T_n(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Defined by :math:`T_n(\cos\theta) = \cos(n\theta)` and built by the
    three-term recurrence

    .. math::

        T_0(x) = 1, \quad T_1(x) = x, \quad
        T_{k+1}(x) = 2x\, T_k(x) - T_{k-1}(x).

    On :math:`[-1, 1]`, :math:`|T_n(x)| \le 1`; the extrema lie at the
    Chebyshev nodes :math:`\cos(j\pi/n)` for :math:`j = 0,\ldots,n`.
    Raises ``ValueError`` for ``n < 0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import chebyshev_polynomial_t
    >>> chebyshev_polynomial_t(lucid.tensor([-1.0, 0.0, 0.5, 1.0]), n=3)
    Tensor([-1.0000, 0.0000, -0.5000, 1.0000])
    """
    p0 = lucid.ones_like(x)
    p1 = x
    return _ortho_poly(x, n, p0, p1, lambda p, q, _k: 2 * x * p - q)


def chebyshev_polynomial_u(x: Tensor, n: int) -> Tensor:
    r"""Chebyshev polynomial of the second kind, :math:`U_n(x)`.

    The U polynomials are orthogonal on :math:`[-1, 1]` with weight
    :math:`\sqrt{1 - x^2}`.  They appear naturally as the derivative of
    the T polynomials (up to a normalisation) and in Gauss-Chebyshev
    quadrature of the second kind.

    Parameters
    ----------
    x : Tensor
        Argument tensor; any floating-point dtype.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`U_n(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Defined by :math:`U_n(\cos\theta) = \sin((n+1)\theta)/\sin\theta`
    and obeying

    .. math::

        U_0(x) = 1, \quad U_1(x) = 2x, \quad
        U_{k+1}(x) = 2x\, U_k(x) - U_{k-1}(x).

    Linked to T by :math:`T_n'(x) = n\, U_{n-1}(x)`.  Raises
    ``ValueError`` for ``n < 0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import chebyshev_polynomial_u
    >>> chebyshev_polynomial_u(lucid.tensor([-1.0, 0.0, 0.5, 1.0]), n=3)
    Tensor([-4.0000, 0.0000, 0.0000, 4.0000])
    """
    p0 = lucid.ones_like(x)
    p1 = 2 * x
    return _ortho_poly(x, n, p0, p1, lambda p, q, _k: 2 * x * p - q)


def chebyshev_polynomial_v(x: Tensor, n: int) -> Tensor:
    r"""Chebyshev polynomial of the third kind, :math:`V_n(x)`.

    The V polynomials are orthogonal on :math:`[-1, 1]` with weight
    :math:`\sqrt{(1 + x)/(1 - x)}` and arise in problems with one
    Dirichlet and one Neumann boundary condition along the interval.

    Parameters
    ----------
    x : Tensor
        Argument tensor; any floating-point dtype.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`V_n(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Half-angle definition

    .. math::

        V_n(\cos\theta) = \frac{\cos((n + \tfrac{1}{2})\theta)}
                                {\cos(\theta/2)},

    with the recurrence

    .. math::

        V_0(x) = 1, \quad V_1(x) = 2x - 1, \quad
        V_{k+1}(x) = 2x\, V_k(x) - V_{k-1}(x).

    Raises ``ValueError`` for ``n < 0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import chebyshev_polynomial_v
    >>> chebyshev_polynomial_v(lucid.tensor([-1.0, 0.0, 1.0]), n=2)
    Tensor([-3.0000, -1.0000, 1.0000])
    """
    p0 = lucid.ones_like(x)
    p1 = 2 * x - 1
    return _ortho_poly(x, n, p0, p1, lambda p, q, _k: 2 * x * p - q)


def chebyshev_polynomial_w(x: Tensor, n: int) -> Tensor:
    r"""Chebyshev polynomial of the fourth kind, :math:`W_n(x)`.

    The W polynomials are orthogonal on :math:`[-1, 1]` with weight
    :math:`\sqrt{(1 - x)/(1 + x)}` — the mirror image of the V family —
    and partner with V under the substitution :math:`x \mapsto -x`.

    Parameters
    ----------
    x : Tensor
        Argument tensor; any floating-point dtype.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`W_n(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Half-angle definition

    .. math::

        W_n(\cos\theta) = \frac{\sin((n + \tfrac{1}{2})\theta)}
                                {\sin(\theta/2)},

    with the recurrence

    .. math::

        W_0(x) = 1, \quad W_1(x) = 2x + 1, \quad
        W_{k+1}(x) = 2x\, W_k(x) - W_{k-1}(x).

    Symmetry relation: :math:`W_n(-x) = (-1)^n V_n(x)`.  Raises
    ``ValueError`` for ``n < 0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import chebyshev_polynomial_w
    >>> chebyshev_polynomial_w(lucid.tensor([-1.0, 0.0, 1.0]), n=2)
    Tensor([1.0000, -1.0000, 3.0000])
    """
    p0 = lucid.ones_like(x)
    p1 = 2 * x + 1
    return _ortho_poly(x, n, p0, p1, lambda p, q, _k: 2 * x * p - q)


def shifted_chebyshev_polynomial_t(x: Tensor, n: int) -> Tensor:
    r"""Shifted Chebyshev T polynomial :math:`T^*_n(x)` on :math:`[0, 1]`.

    Maps the standard Chebyshev T family from :math:`[-1, 1]` onto the
    unit interval :math:`[0, 1]` via the linear substitution
    :math:`x \mapsto 2x - 1`.  Useful when the natural domain of an
    approximation problem is :math:`[0, 1]`.

    Parameters
    ----------
    x : Tensor
        Argument on the shifted domain :math:`[0, 1]`; any
        floating-point dtype.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`T^*_n(x) = T_n(2x - 1)`, element-wise, same shape and
        dtype as ``x``.

    Notes
    -----
    Definition: :math:`T^*_n(x) = T_n(2x - 1)`.  Inherits orthogonality
    on :math:`[0, 1]` with weight :math:`1/\sqrt{x - x^2}`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import shifted_chebyshev_polynomial_t
    >>> shifted_chebyshev_polynomial_t(lucid.tensor([0.0, 0.5, 1.0]), n=2)
    Tensor([1.0000, -1.0000, 1.0000])
    """
    return chebyshev_polynomial_t(2 * x - 1, n)


def shifted_chebyshev_polynomial_u(x: Tensor, n: int) -> Tensor:
    r"""Shifted Chebyshev U polynomial :math:`U^*_n(x)` on :math:`[0, 1]`.

    Standard Chebyshev U polynomials shifted from :math:`[-1, 1]` onto
    the unit interval by :math:`x \mapsto 2x - 1`.

    Parameters
    ----------
    x : Tensor
        Argument on :math:`[0, 1]`; any floating-point dtype.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`U^*_n(x) = U_n(2x - 1)`, element-wise.

    Notes
    -----
    :math:`U^*_n(x) = U_n(2x - 1)`.  Orthogonal on :math:`[0, 1]` with
    weight :math:`\sqrt{x - x^2}`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import shifted_chebyshev_polynomial_u
    >>> shifted_chebyshev_polynomial_u(lucid.tensor([0.0, 0.5, 1.0]), n=2)
    Tensor([1.0000, -1.0000, 1.0000])
    """
    return chebyshev_polynomial_u(2 * x - 1, n)


def shifted_chebyshev_polynomial_v(x: Tensor, n: int) -> Tensor:
    r"""Shifted Chebyshev V polynomial :math:`V^*_n(x)` on :math:`[0, 1]`.

    Chebyshev V polynomials of the third kind composed with
    :math:`x \mapsto 2x - 1`, yielding a basis orthogonal on
    :math:`[0, 1]`.

    Parameters
    ----------
    x : Tensor
        Argument on :math:`[0, 1]`; any floating-point dtype.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`V^*_n(x) = V_n(2x - 1)`, element-wise.

    Notes
    -----
    :math:`V^*_n(x) = V_n(2x - 1)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import shifted_chebyshev_polynomial_v
    >>> shifted_chebyshev_polynomial_v(lucid.tensor([0.0, 1.0]), n=1)
    Tensor([-3.0000, 1.0000])
    """
    return chebyshev_polynomial_v(2 * x - 1, n)


def shifted_chebyshev_polynomial_w(x: Tensor, n: int) -> Tensor:
    r"""Shifted Chebyshev W polynomial :math:`W^*_n(x)` on :math:`[0, 1]`.

    Chebyshev W polynomials of the fourth kind composed with
    :math:`x \mapsto 2x - 1`.

    Parameters
    ----------
    x : Tensor
        Argument on :math:`[0, 1]`; any floating-point dtype.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`W^*_n(x) = W_n(2x - 1)`, element-wise.

    Notes
    -----
    :math:`W^*_n(x) = W_n(2x - 1)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import shifted_chebyshev_polynomial_w
    >>> shifted_chebyshev_polynomial_w(lucid.tensor([0.0, 1.0]), n=1)
    Tensor([-1.0000, 3.0000])
    """
    return chebyshev_polynomial_w(2 * x - 1, n)


def hermite_polynomial_h(x: Tensor, n: int) -> Tensor:
    r"""Physicist's Hermite polynomial :math:`H_n(x)`.

    The physicists' Hermite polynomials are orthogonal on
    :math:`(-\infty, \infty)` with weight :math:`e^{-x^2}` and form the
    eigenfunctions of the quantum harmonic oscillator (in conjunction
    with the Gaussian envelope).

    Parameters
    ----------
    x : Tensor
        Argument tensor; any floating-point dtype.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`H_n(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Rodrigues formula and recurrence:

    .. math::

        H_n(x) = (-1)^n e^{x^2}
                 \frac{d^n}{dx^n} e^{-x^2}, \qquad
        H_{k+1}(x) = 2x\, H_k(x) - 2k\, H_{k-1}(x).

    Starting values: :math:`H_0(x) = 1`, :math:`H_1(x) = 2x`.  Linked to
    the probabilists' Hermite polynomials by :math:`H_n(x) =
    2^{n/2} He_n(\sqrt{2}\, x)`.  Raises ``ValueError`` for ``n < 0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import hermite_polynomial_h
    >>> hermite_polynomial_h(lucid.tensor([-1.0, 0.0, 1.0]), n=3)
    Tensor([4.0000, 0.0000, -4.0000])
    """
    p0 = lucid.ones_like(x)
    p1 = 2 * x
    return _ortho_poly(
        x,
        n,
        p0,
        p1,
        lambda p, q, k: 2 * x * p - 2 * float(k) * q,
    )


def hermite_polynomial_he(x: Tensor, n: int) -> Tensor:
    r"""Probabilist's Hermite polynomial :math:`\mathit{He}_n(x)`.

    Probabilists' Hermite polynomials are orthogonal on
    :math:`(-\infty, \infty)` with weight :math:`e^{-x^2/2} /
    \sqrt{2\pi}` — the standard normal density — and are the natural
    basis for Hermite-expansions of functions against a Gaussian
    measure (Wiener chaos, Gauss-Hermite quadrature with weight 1).

    Parameters
    ----------
    x : Tensor
        Argument tensor; any floating-point dtype.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`\mathit{He}_n(x)` element-wise, same shape and dtype as
        ``x``.

    Notes
    -----
    Recurrence:

    .. math::

        \mathit{He}_0(x) = 1, \quad \mathit{He}_1(x) = x, \quad
        \mathit{He}_{k+1}(x) = x\, \mathit{He}_k(x)
                             - k\, \mathit{He}_{k-1}(x).

    Connection to physicists' Hermite: :math:`\mathit{He}_n(x) =
    2^{-n/2} H_n(x/\sqrt 2)`.  Raises ``ValueError`` for ``n < 0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import hermite_polynomial_he
    >>> hermite_polynomial_he(lucid.tensor([-1.0, 0.0, 1.0]), n=3)
    Tensor([2.0000, 0.0000, -2.0000])
    """
    p0 = lucid.ones_like(x)
    p1 = x
    return _ortho_poly(
        x,
        n,
        p0,
        p1,
        lambda p, q, k: x * p - float(k) * q,
    )


def legendre_polynomial_p(x: Tensor, n: int) -> Tensor:
    r"""Legendre polynomial :math:`P_n(x)`.

    The Legendre polynomials are orthogonal on :math:`[-1, 1]` with
    uniform weight, and serve as the angular eigenfunctions of the
    Laplacian on the sphere (spherical harmonics with :math:`m = 0`)
    and as a basis for Gauss-Legendre quadrature.

    Parameters
    ----------
    x : Tensor
        Argument tensor; any floating-point dtype.  Customarily
        evaluated on :math:`[-1, 1]`.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`P_n(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Bonnet recurrence:

    .. math::

        P_0(x) = 1, \quad P_1(x) = x, \quad
        (k+1)\, P_{k+1}(x) = (2k + 1)\, x\, P_k(x) - k\, P_{k-1}(x).

    On :math:`[-1, 1]`, :math:`|P_n(x)| \le 1` with boundary values
    :math:`P_n(1) = 1`, :math:`P_n(-1) = (-1)^n`.  Raises
    ``ValueError`` for ``n < 0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import legendre_polynomial_p
    >>> legendre_polynomial_p(lucid.tensor([-1.0, 0.0, 0.5, 1.0]), n=3)
    Tensor([-1.0000, 0.0000, -0.4375, 1.0000])
    """
    p0 = lucid.ones_like(x)
    p1 = x
    return _ortho_poly(
        x,
        n,
        p0,
        p1,
        lambda p, q, k: ((2 * k + 1) * x * p - float(k) * q) / float(k + 1),
    )


def laguerre_polynomial_l(x: Tensor, n: int) -> Tensor:
    r"""Laguerre polynomial :math:`L_n(x)`.

    The (simple) Laguerre polynomials are orthogonal on
    :math:`[0, \infty)` with weight :math:`e^{-x}` and appear as the
    radial eigenfunctions of the hydrogen atom, as the basis of
    Gauss-Laguerre quadrature, and in survival-analysis kernels.

    Parameters
    ----------
    x : Tensor
        Argument tensor; any floating-point dtype.  Customarily
        evaluated on :math:`[0, \infty)`.
    n : int
        Non-negative polynomial degree.

    Returns
    -------
    Tensor
        :math:`L_n(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Recurrence:

    .. math::

        L_0(x) = 1, \quad L_1(x) = 1 - x, \quad
        (k+1)\, L_{k+1}(x) = (2k + 1 - x)\, L_k(x)
                           - k\, L_{k-1}(x).

    The leading coefficient is :math:`(-x)^n / n!`, and
    :math:`L_n(0) = 1` for every ``n``.  Raises ``ValueError`` for
    ``n < 0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import laguerre_polynomial_l
    >>> laguerre_polynomial_l(lucid.tensor([0.0, 1.0, 2.0]), n=2)
    Tensor([1.0000, -0.5000, -1.0000])
    """
    p0 = lucid.ones_like(x)
    p1 = 1 - x
    return _ortho_poly(
        x,
        n,
        p0,
        p1,
        lambda p, q, k: ((2 * k + 1 - x) * p - float(k) * q) / float(k + 1),
    )


# ── Bessel functions ───────────────────────────────────────────────────────
#
# Polynomial / asymptotic approximations from Abramowitz & Stegun §9
# (J0, J1, Y0, Y1) and §9.8 (K0, K1).  Each is split into a small-|x|
# polynomial branch and a large-|x| asymptotic-with-trig branch.  These
# are the same approximations used by Cephes / SciPy and are accurate to
# ≈ 1e-7 over the supported range, which is sufficient for typical ML
# usage.  Higher precision needs would call into a proper implementation
# (mpmath / Boost.Math).


def _split(
    x: Tensor,
    threshold: float,
    small_fn: Callable[[Tensor], Tensor],
    large_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """``where(|x| <= threshold, small_fn(x), large_fn(x))`` evaluated
    safely on both branches by always running both.  Branches are
    expected to be regular (no NaN / inf) on their respective domains."""
    mask = lucid.abs(x) <= threshold
    return lucid.where(mask, small_fn(x), large_fn(x))


def _bessel_j0_small(x: Tensor) -> Tensor:
    # Abramowitz & Stegun 9.4.1 — polynomial fit on |x| <= 3.
    y = (x / 3.0) ** 2
    return (
        1.0
        - 2.2499997 * y
        + 1.2656208 * y**2
        - 0.3163866 * y**3
        + 0.0444479 * y**4
        - 0.0039444 * y**5
        + 0.00021 * y**6
    )


def _bessel_j0_large(x: Tensor) -> Tensor:
    # Abramowitz & Stegun 9.4.3 — asymptotic on |x| > 3.
    ax = lucid.abs(x)
    z = 3.0 / ax
    f0 = (
        0.79788456
        - 0.00000077 * z
        - 0.00552740 * z**2
        - 0.00009512 * z**3
        + 0.00137237 * z**4
        - 0.00072805 * z**5
        + 0.00014476 * z**6
    )
    theta = (
        ax
        - 0.78539816
        - 0.04166397 * z
        - 0.00003954 * z**2
        + 0.00262573 * z**3
        - 0.00054125 * z**4
        - 0.00029333 * z**5
        + 0.00013558 * z**6
    )
    return f0 * lucid.cos(theta) / lucid.sqrt(ax)


def bessel_j0(x: Tensor) -> Tensor:
    r"""Bessel function of the first kind, order 0.

    Computes :math:`J_0(x)`, the regular-at-origin solution of Bessel's
    equation :math:`x^2 y'' + x y' + x^2 y = 0` for order 0.  Appears
    throughout wave propagation, optical diffraction (the Airy disk
    profile), and vibration of circular membranes.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`J_0(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Series representation:

    .. math::

        J_0(x) = \sum_{k=0}^\infty \frac{(-1)^k}{(k!)^2}
                 \left(\frac{x}{2}\right)^{2k}.

    The implementation uses the Abramowitz & Stegun §9.4 two-branch
    polynomial fit: a power series in :math:`(x/3)^2` for
    :math:`|x| \le 3` and an asymptotic
    :math:`\sqrt{2/(\pi x)} \cos(x - \pi/4) \cdot f(3/|x|)` form for
    :math:`|x| > 3`.  Accuracy is :math:`\approx 10^{-7}` on the real
    line.  ``J_0`` is even, :math:`J_0(0) = 1`, and the function decays
    as :math:`\sqrt{2/(\pi x)}` for large ``x``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import bessel_j0
    >>> bessel_j0(lucid.tensor([0.0, 1.0, 2.4048, 5.0]))
    Tensor([1.0000, 0.7652, 0.0000, -0.1776])
    """
    return _split(x, 3.0, _bessel_j0_small, _bessel_j0_large)


def _bessel_j1_small(x: Tensor) -> Tensor:
    y = (x / 3.0) ** 2
    poly = (
        0.5
        - 0.56249985 * y
        + 0.21093573 * y**2
        - 0.03954289 * y**3
        + 0.00443319 * y**4
        - 0.00031761 * y**5
        + 0.00001109 * y**6
    )
    return x * poly


def _bessel_j1_large(x: Tensor) -> Tensor:
    ax = lucid.abs(x)
    z = 3.0 / ax
    f1 = (
        0.79788456
        + 0.00000156 * z
        + 0.01659667 * z**2
        + 0.00017105 * z**3
        - 0.00249511 * z**4
        + 0.00113653 * z**5
        - 0.00020033 * z**6
    )
    theta = (
        ax
        - 2.35619449
        + 0.12499612 * z
        + 0.00005650 * z**2
        - 0.00637879 * z**3
        + 0.00074348 * z**4
        + 0.00079824 * z**5
        - 0.00029166 * z**6
    )
    val = f1 * lucid.cos(theta) / lucid.sqrt(ax)
    # j1 is odd in x; flip sign for negative inputs.
    return lucid.where(x < 0, -val, val)


def bessel_j1(x: Tensor) -> Tensor:
    r"""Bessel function of the first kind, order 1.

    Computes :math:`J_1(x)`, the regular order-one solution of Bessel's
    equation.  Closely related to :math:`J_0` by
    :math:`J_0'(x) = -J_1(x)`; appears as the radial derivative of
    cylindrical waves and in the antenna-pattern of a uniformly
    illuminated circular aperture.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`J_1(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Series representation:

    .. math::

        J_1(x) = \sum_{k=0}^\infty \frac{(-1)^k}{k!\, (k+1)!}
                 \left(\frac{x}{2}\right)^{2k+1}.

    Implementation: Abramowitz & Stegun §9.4 two-branch polynomial,
    accurate to :math:`\approx 10^{-7}`.  ``J_1`` is odd,
    :math:`J_1(0) = 0`, with maximum :math:`\approx 0.5819` at
    :math:`x \approx 1.8412`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import bessel_j1
    >>> bessel_j1(lucid.tensor([0.0, 1.0, 3.8317]))
    Tensor([0.0000, 0.4401, 0.0000])
    """
    return _split(x, 3.0, _bessel_j1_small, _bessel_j1_large)


_TWO_OVER_PI = 2.0 / math.pi


# Y0 / Y1 use the rational approximations from Numerical Recipes (Press,
# Teukolsky, Vetterling, Flannery — 2nd ed., §6.5).  These are carefully
# fitted on (0, 8] with the J0 / J1·log(x) Wronskian term added on top,
# and on (8, ∞) via the standard A&S 9.4.5 trigonometric asymptotic.


def _bessel_y0_small(x: Tensor) -> Tensor:
    safe_x = lucid.where(x == lucid.zeros_like(x), lucid.ones_like(x), x)
    y = x * x
    num = -2957821389.0 + y * (
        7062834065.0
        + y * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733)))
    )
    den = 40076544269.0 + y * (
        745249964.8
        + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0)))
    )
    return num / den + _TWO_OVER_PI * lucid.log(safe_x) * bessel_j0(x)


def _bessel_y0_large(x: Tensor) -> Tensor:
    z = 8.0 / x
    y = z * z
    p = 1.0 + y * (
        -0.1098628627e-2
        + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6))
    )
    q = -0.1562499995e-1 + y * (
        0.1430488765e-3
        + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934935152e-7)))
    )
    xx = x - 0.785398164
    return lucid.sqrt(0.636619772 / x) * (lucid.sin(xx) * p + z * lucid.cos(xx) * q)


def bessel_y0(x: Tensor) -> Tensor:
    r"""Bessel function of the second kind, order 0.

    Computes :math:`Y_0(x)`, the singular-at-origin partner of
    :math:`J_0`.  Together :math:`J_0` and :math:`Y_0` span the
    solution space of Bessel's equation at order 0; :math:`Y_0` is the
    one that diverges (logarithmically) at the origin.

    Parameters
    ----------
    x : Tensor
        Input tensor; only :math:`x > 0` is in the domain.  Any
        floating-point dtype.

    Returns
    -------
    Tensor
        :math:`Y_0(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Asymptotic behaviour:

    .. math::

        Y_0(x) \sim \frac{2}{\pi}\left(\log\frac{x}{2}
                                       + \gamma\right) \;
                                 \text{as } x \to 0^+,
        \qquad
        Y_0(x) \sim \sqrt{\frac{2}{\pi x}}
                   \sin\!\left(x - \frac{\pi}{4}\right)
                                 \;\text{as } x \to \infty.

    Implementation: Numerical Recipes §6.5 rational fit on
    :math:`(0, 8]` with the :math:`J_0 \log` Wronskian term, and the
    standard A&S 9.4.5 trigonometric asymptotic on :math:`(8, \infty)`.
    Accuracy is :math:`\approx 10^{-7}`.  ``Y_0`` diverges to
    :math:`-\infty` as :math:`x \to 0^+`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import bessel_y0
    >>> bessel_y0(lucid.tensor([1.0, 5.0, 10.0]))
    Tensor([0.0883, -0.3085, 0.0557])
    """
    return _split(x, 8.0, _bessel_y0_small, _bessel_y0_large)


def _bessel_y1_small(x: Tensor) -> Tensor:
    safe_x = lucid.where(x == lucid.zeros_like(x), lucid.ones_like(x), x)
    y = x * x
    num = x * (
        -0.4900604943e13
        + y
        * (
            0.1275274390e13
            + y
            * (
                -0.5153438139e11
                + y * (0.7349264551e9 + y * (-0.4237922726e7 + y * 0.8511937935e4))
            )
        )
    )
    den = 0.2499580570e14 + y * (
        0.4244419664e12
        + y
        * (
            0.3733650367e10
            + y
            * (0.2245904002e8 + y * (0.1020426050e6 + y * (0.3549632885e3 + y * 1.0)))
        )
    )
    return num / den + _TWO_OVER_PI * (lucid.log(safe_x) * bessel_j1(x) - 1.0 / safe_x)


def _bessel_y1_large(x: Tensor) -> Tensor:
    z = 8.0 / x
    y = z * z
    p = 1.0 + y * (
        0.183105e-2
        + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6)))
    )
    q = 0.04687499995 + y * (
        -0.2002690873e-3
        + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6))
    )
    xx = x - 2.356194491
    return lucid.sqrt(0.636619772 / x) * (lucid.sin(xx) * p + z * lucid.cos(xx) * q)


def bessel_y1(x: Tensor) -> Tensor:
    r"""Bessel function of the second kind, order 1.

    Computes :math:`Y_1(x)`, the singular-at-origin order-one solution
    of Bessel's equation.  Diverges as :math:`-2/(\pi x)` at the origin
    and decays as :math:`\sqrt{2/(\pi x)}` for large ``x``.

    Parameters
    ----------
    x : Tensor
        Input tensor; only :math:`x > 0` is in the domain.  Any
        floating-point dtype.

    Returns
    -------
    Tensor
        :math:`Y_1(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Asymptotic forms:

    .. math::

        Y_1(x) \sim -\frac{2}{\pi x} \;\text{as } x \to 0^+,
        \qquad
        Y_1(x) \sim \sqrt{\frac{2}{\pi x}}
                    \sin\!\left(x - \frac{3\pi}{4}\right)
                    \;\text{as } x \to \infty.

    Implementation: Numerical Recipes §6.5 rational fit on
    :math:`(0, 8]` (Wronskian-corrected with the :math:`J_1 \log`
    term), trigonometric asymptotic on :math:`(8, \infty)`.  Accuracy
    is :math:`\approx 10^{-7}`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import bessel_y1
    >>> bessel_y1(lucid.tensor([1.0, 5.0, 10.0]))
    Tensor([-0.7812, 0.1479, 0.2490])
    """
    return _split(x, 8.0, _bessel_y1_small, _bessel_y1_large)


def _modified_bessel_k0_small(x: Tensor) -> Tensor:
    # A&S 9.8.5 — polynomial fit for 0 < x <= 2.
    y = (x / 2.0) ** 2
    safe_x = lucid.where(x == lucid.zeros_like(x), lucid.ones_like(x), x)
    poly = (
        -0.57721566
        + 0.42278420 * y
        + 0.23069756 * y**2
        + 0.03488590 * y**3
        + 0.00262698 * y**4
        + 0.00010750 * y**5
        + 0.00000740 * y**6
    )
    return -lucid.log(safe_x / 2.0) * lucid.i0(x) + poly


def _modified_bessel_k0_large(x: Tensor) -> Tensor:
    # A&S 9.8.6 — asymptotic for x > 2.
    z = 2.0 / x
    poly = (
        1.25331414
        - 0.07832358 * z
        + 0.02189568 * z**2
        - 0.01062446 * z**3
        + 0.00587872 * z**4
        - 0.00251540 * z**5
        + 0.00053208 * z**6
    )
    return poly * lucid.exp(-x) / lucid.sqrt(x)


def modified_bessel_k0(x: Tensor) -> Tensor:
    r"""Modified Bessel function of the second kind, order 0.

    Computes :math:`K_0(x)`, the exponentially decaying solution of the
    modified Bessel equation at order 0.  Appears as the Green's
    function of the 2D Helmholtz operator and as the log-density kernel
    of certain heavy-tailed distributions.

    Parameters
    ----------
    x : Tensor
        Input tensor; only :math:`x > 0` is in the domain.  Any
        floating-point dtype.

    Returns
    -------
    Tensor
        :math:`K_0(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Asymptotic forms:

    .. math::

        K_0(x) \sim -\log(x/2) - \gamma \;\text{as } x \to 0^+,
        \qquad
        K_0(x) \sim \sqrt{\frac{\pi}{2x}}\, e^{-x}
                  \;\text{as } x \to \infty.

    Implementation: Abramowitz & Stegun §9.8 polynomial branches at the
    cutoff :math:`x = 2`, accurate to :math:`\approx 10^{-7}`.  For
    numerically stable evaluation at large arguments use
    :func:`scaled_modified_bessel_k0`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import modified_bessel_k0
    >>> modified_bessel_k0(lucid.tensor([0.5, 1.0, 5.0]))
    Tensor([0.9244, 0.4210, 0.0037])
    """
    return _split(x, 2.0, _modified_bessel_k0_small, _modified_bessel_k0_large)


def scaled_modified_bessel_k0(x: Tensor) -> Tensor:
    r"""Exponentially scaled modified Bessel :math:`K_0`: :math:`e^x K_0(x)`.

    Computes :math:`e^{x} K_0(x)`, the bounded counterpart of
    :func:`modified_bessel_k0`.  Because :math:`K_0` decays
    exponentially as :math:`e^{-x}/\sqrt{x}`, the scaled form is the
    natural quantity to manipulate in log-density expressions where
    direct evaluation of :math:`K_0` would underflow.

    Parameters
    ----------
    x : Tensor
        Input tensor; only :math:`x > 0` is in the domain.

    Returns
    -------
    Tensor
        :math:`e^x K_0(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Asymptotic: :math:`e^x K_0(x) \sim \sqrt{\pi/(2x)}` as
    :math:`x \to \infty`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import scaled_modified_bessel_k0
    >>> scaled_modified_bessel_k0(lucid.tensor([1.0, 5.0, 20.0]))
    Tensor([1.1445, 0.5478, 0.2745])
    """
    return modified_bessel_k0(x) * lucid.exp(x)


def _modified_bessel_k1_small(x: Tensor) -> Tensor:
    y = (x / 2.0) ** 2
    safe_x = lucid.where(x == lucid.zeros_like(x), lucid.ones_like(x), x)
    poly = (
        1.0
        + 0.15443144 * y
        - 0.67278579 * y**2
        - 0.18156897 * y**3
        - 0.01919402 * y**4
        - 0.00110404 * y**5
        - 0.00004686 * y**6
    )
    return lucid.log(safe_x / 2.0) * i1(x) + poly / safe_x


def _modified_bessel_k1_large(x: Tensor) -> Tensor:
    z = 2.0 / x
    poly = (
        1.25331414
        + 0.23498619 * z
        - 0.03655620 * z**2
        + 0.01504268 * z**3
        - 0.00780353 * z**4
        + 0.00325614 * z**5
        - 0.00068245 * z**6
    )
    return poly * lucid.exp(-x) / lucid.sqrt(x)


def modified_bessel_k1(x: Tensor) -> Tensor:
    r"""Modified Bessel function of the second kind, order 1.

    Computes :math:`K_1(x)`, the order-one analogue of :math:`K_0`.
    Diverges as :math:`1/x` near the origin and decays exponentially
    for large argument.  Appears in Generalised-Inverse-Gaussian and
    Normal-Inverse-Gaussian log-densities.

    Parameters
    ----------
    x : Tensor
        Input tensor; only :math:`x > 0` is in the domain.

    Returns
    -------
    Tensor
        :math:`K_1(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Asymptotic forms:

    .. math::

        K_1(x) \sim \frac{1}{x} \;\text{as } x \to 0^+,
        \qquad
        K_1(x) \sim \sqrt{\frac{\pi}{2x}}\, e^{-x}
                  \;\text{as } x \to \infty.

    Implementation: Abramowitz & Stegun §9.8 two-branch polynomial,
    accurate to :math:`\approx 10^{-7}`.  Use
    :func:`scaled_modified_bessel_k1` for numerically stable
    evaluation at large ``x``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import modified_bessel_k1
    >>> modified_bessel_k1(lucid.tensor([0.5, 1.0, 5.0]))
    Tensor([1.6564, 0.6019, 0.0040])
    """
    return _split(x, 2.0, _modified_bessel_k1_small, _modified_bessel_k1_large)


def scaled_modified_bessel_k1(x: Tensor) -> Tensor:
    r"""Exponentially scaled modified Bessel :math:`K_1`: :math:`e^x K_1(x)`.

    Computes :math:`e^x K_1(x)`, the bounded scaled variant of
    :func:`modified_bessel_k1` suitable for stable evaluation of
    log-densities involving :math:`K_1` at large argument.

    Parameters
    ----------
    x : Tensor
        Input tensor; only :math:`x > 0` is in the domain.

    Returns
    -------
    Tensor
        :math:`e^x K_1(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Asymptotic: :math:`e^x K_1(x) \sim \sqrt{\pi/(2x)}` as
    :math:`x \to \infty`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import scaled_modified_bessel_k1
    >>> scaled_modified_bessel_k1(lucid.tensor([1.0, 5.0, 20.0]))
    Tensor([1.6362, 0.6001, 0.2820])
    """
    return modified_bessel_k1(x) * lucid.exp(x)


# ── Hurwitz zeta ───────────────────────────────────────────────────────────


def zeta(x: Tensor, q: Tensor) -> Tensor:
    r"""Hurwitz zeta function :math:`\zeta(x, q)`.

    Computes the Hurwitz zeta function, a two-argument generalisation
    of the Riemann zeta that arises in moment-generating functions of
    discrete distributions, in expansions of polygamma functions, and
    as a regulariser in analytic number theory.

    Parameters
    ----------
    x : Tensor
        Exponent; must satisfy :math:`\Re(x) > 1` for the series to
        converge.  Any floating-point dtype.
    q : Tensor
        Shift parameter; broadcast-compatible with ``x``.  Should be
        positive (avoid the poles at non-positive integers).

    Returns
    -------
    Tensor
        :math:`\zeta(x, q)` element-wise, broadcast to the common shape
        of ``x`` and ``q``.

    Notes
    -----
    Series definition:

    .. math::

        \zeta(x, q) = \sum_{k=0}^\infty \frac{1}{(k + q)^x}.

    The implementation accumulates an explicit prefix of 12 terms and
    closes the tail with the Euler–Maclaurin correction

    .. math::

        \sum_{k=N}^\infty (k + q)^{-x}
            \approx \frac{a^{1-x}}{x - 1}
                  + \frac{a^{-x}}{2}
                  + \sum_j \frac{B_{2j}}{(2j)!}
                    \prod_{i=0}^{2j-2}(x + i)\, a^{-x-2j+1},

    where :math:`a = q + N` and the :math:`B_{2j}` are Bernoulli
    numbers.  Accuracy is roughly :math:`10^{-6}` for :math:`x > 1`
    and moderate ``q`` — adequate for ML log-density work, not for
    heroic-precision scientific computing.  The Riemann zeta is the
    special case :math:`\zeta(x) = \zeta(x, 1)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import zeta
    >>> x = lucid.tensor([2.0, 3.0])
    >>> q = lucid.tensor([1.0, 1.0])
    >>> zeta(x, q)
    Tensor([1.6449, 1.2021])
    """
    # Accumulate the explicit prefix Σ_{k=0..N-1} (k + q)^{-x}.
    N = 12
    s = lucid.zeros_like(q)
    for k in range(N):
        s = s + (q + float(k)) ** (-x)

    # Euler-Maclaurin tail starting at a = q + N:
    #   Σ_{k=N..∞} (k+q)^{-x}  ≈  a^{1-x} / (x - 1)  +  ½ · a^{-x}
    #                            +  Σ_j  B_{2j} / (2j)! · (Π_{i=0..2j-2} (x + i)) · a^{-x - 2j + 1}
    a = q + float(N)
    one = lucid.ones_like(x)
    head = a ** (1 - x) / (x - one)
    half = 0.5 * a ** (-x)

    bernoulli = [
        1.0 / 6.0,  # B_2
        -1.0 / 30.0,  # B_4
        1.0 / 42.0,  # B_6
        -1.0 / 30.0,  # B_8
        5.0 / 66.0,  # B_10
    ]
    factorial = [2.0, 24.0, 720.0, 40320.0, 3628800.0]
    tail = lucid.zeros_like(s)
    coef = lucid.ones_like(x)
    for j, (b, f) in enumerate(zip(bernoulli, factorial)):
        # coef accumulates Π_{i=0..2j} (x + i).
        coef = coef * (x + float(2 * j)) * (x + float(2 * j + 1))
        tail = tail + (b / f) * coef * a ** (-x - float(2 * j + 1))

    return s + head + half + tail


# ── Name aliases matching reference framework's lucid.special surface ─────────


def gammaln(x: Tensor) -> Tensor:
    r"""Natural logarithm of the absolute value of the gamma function.

    Convenience alias for :func:`lucid.lgamma`.  Computes
    :math:`\log|\Gamma(x)|` with numerical stability for large
    arguments where :math:`\Gamma(x)` would itself overflow.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\log|\Gamma(x)|` element-wise.

    Notes
    -----
    Definition:

    .. math::

        \log|\Gamma(x)| = \log\!\left|
            \int_0^\infty t^{x-1} e^{-t}\, dt \right|.

    For positive integer ``n``, :math:`\Gamma(n) = (n - 1)!`, so
    ``gammaln(n) = log((n-1)!)``.  ``gammaln`` returns ``+inf`` at
    non-positive integers (poles of the gamma function).

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import gammaln
    >>> gammaln(lucid.tensor([1.0, 5.0, 100.0]))
    Tensor([0.0000, 3.1781, 359.1342])
    """
    return lucid.lgamma(x)


def psi(x: Tensor) -> Tensor:
    r"""Digamma function :math:`\psi(x) = \frac{d}{dx} \log \Gamma(x)`.

    Convenience alias for :func:`lucid.digamma`.  The digamma function
    is the logarithmic derivative of the gamma function and the
    fundamental building block of the polygamma family.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\psi(x)` element-wise.

    Notes
    -----
    Definition:

    .. math::

        \psi(x) = \frac{d}{dx} \log \Gamma(x)
                = \frac{\Gamma'(x)}{\Gamma(x)}
                = -\gamma + \sum_{k=0}^\infty
                  \left(\frac{1}{k + 1} - \frac{1}{k + x}\right),

    where :math:`\gamma \approx 0.5772` is the Euler–Mascheroni
    constant.  Used in the derivatives of log-likelihoods of the
    Gamma / Beta / Dirichlet distributions.  Has simple poles at the
    non-positive integers.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import psi
    >>> psi(lucid.tensor([1.0, 2.0, 10.0]))
    Tensor([-0.5772, 0.4228, 2.2517])
    """
    return lucid.digamma(x)


def expit(x: Tensor) -> Tensor:
    r"""Logistic sigmoid (expit) function :math:`\sigma(x)`.

    Convenience alias for :func:`lucid.sigmoid`.  Maps the real line
    onto the open interval :math:`(0, 1)` and is the inverse of
    :func:`lucid.logit`.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\sigma(x) = 1 / (1 + e^{-x})` element-wise, same shape
        and dtype as ``x``.

    Notes
    -----
    Definition:

    .. math::

        \sigma(x) = \frac{1}{1 + e^{-x}}
                  = \frac{e^x}{1 + e^x}.

    Symmetry: :math:`\sigma(-x) = 1 - \sigma(x)`.  Derivative:
    :math:`\sigma'(x) = \sigma(x)\,(1 - \sigma(x))`.  Frequently used
    as the probabilistic-output activation of binary classifiers.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import expit
    >>> expit(lucid.tensor([-2.0, 0.0, 2.0]))
    Tensor([0.1192, 0.5000, 0.8808])
    """
    return lucid.sigmoid(x)


def modified_bessel_i0(x: Tensor) -> Tensor:
    r"""Modified Bessel function of the first kind, order 0.

    Convenience alias for :func:`lucid.i0`.  Computes :math:`I_0(x)`,
    the order-zero regular solution of the modified Bessel equation,
    used in the von Mises density and rotational kernel families.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_0(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Series form:

    .. math::

        I_0(x) = \sum_{k=0}^\infty \frac{1}{(k!)^2}
                 \left(\frac{x}{2}\right)^{2k}.

    Even function; :math:`I_0(0) = 1`; grows as
    :math:`e^{|x|}/\sqrt{2\pi |x|}` for large argument — use
    :func:`i0e` for numerical stability at large ``|x|``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import modified_bessel_i0
    >>> modified_bessel_i0(lucid.tensor([0.0, 1.0, 3.0]))
    Tensor([1.0000, 1.2661, 4.8808])
    """
    return lucid.i0(x)


def modified_bessel_i1(x: Tensor) -> Tensor:
    r"""Modified Bessel function of the first kind, order 1.

    Convenience wrapper around :func:`i1`.  Computes :math:`I_1(x)`,
    the order-one regular solution of the modified Bessel equation,
    forming a natural pair with :math:`I_0` for log-density evaluation
    of Rician / Bessel-family distributions.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_1(x)` element-wise.

    Notes
    -----
    Series form:

    .. math::

        I_1(x) = \sum_{k=0}^\infty \frac{1}{k!\,(k+1)!}
                 \left(\frac{x}{2}\right)^{2k+1}.

    Odd function; :math:`I_1(0) = 0`; grows like :math:`I_0` at
    infinity.  Use :func:`i1e` for stable evaluation at large ``|x|``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import modified_bessel_i1
    >>> modified_bessel_i1(lucid.tensor([0.0, 1.0, 3.0]))
    Tensor([0.0000, 0.5652, 3.9534])
    """
    return i1(x)


__all__ = [
    "erfcx",
    "i0e",
    "i1",
    "i1e",
    "ndtr",
    "ndtri",
    "log_ndtr",
    "xlog1py",
    "entr",
    "multigammaln",
    "polygamma",
    "spherical_bessel_j0",
    # Orthogonal polynomials (12)
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "legendre_polynomial_p",
    "laguerre_polynomial_l",
    # Bessel + modified Bessel K (8)
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "bessel_y1",
    "modified_bessel_k0",
    "modified_bessel_k1",
    "scaled_modified_bessel_k0",
    "scaled_modified_bessel_k1",
    # Hurwitz zeta (1)
    "zeta",
    # Name aliases
    "gammaln",
    "psi",
    "expit",
    "modified_bessel_i0",
    "modified_bessel_i1",
]
