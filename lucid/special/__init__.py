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
from typing import TYPE_CHECKING

import lucid

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Error-function family ──────────────────────────────────────────────────


def erfcx(x: Tensor) -> Tensor:
    """Scaled complementary error function: ``exp(x²) · erfc(x)``.

    For ``x ≫ 0`` the unscaled ``erfc(x)`` underflows quickly while
    ``erfcx`` stays finite; this matters for tail-probability work.
    The straightforward composite is accurate for ``|x| ≲ 5`` and starts
    losing precision for very large positive ``x`` — adequate for general
    use, not for extreme-value statistics.
    """
    return lucid.exp(x * x) * lucid.erfc(x)


# ── Modified Bessel I₀ / I₁ family ─────────────────────────────────────────


def i0e(x: Tensor) -> Tensor:
    """Exponentially scaled ``i0``: ``exp(-|x|) · I₀(x)``.

    Useful when ``i0`` itself overflows (it grows as ``exp(|x|) / √|x|``).
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
    """Modified Bessel function of the first kind, order 1.

    Two-branch Abramowitz polynomial: ``|x| ≤ 3.75`` uses a power series
    in ``(x/3.75)²``; ``|x| > 3.75`` uses an asymptotic series in
    ``3.75/|x|`` multiplied by ``exp(|x|)/√|x|`` and the sign of ``x``.
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
    """Exponentially scaled ``i1``: ``exp(-|x|) · I₁(x)``."""
    return lucid.exp(-lucid.abs(x)) * i1(x)


# ── Normal-distribution helpers ────────────────────────────────────────────

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def ndtr(x: Tensor) -> Tensor:
    """Normal CDF: ``Φ(x) = ½·(1 + erf(x/√2))``."""
    return 0.5 * (1.0 + lucid.erf(x * _INV_SQRT2))


def log_ndtr(x: Tensor) -> Tensor:
    """Numerically stable ``log(Φ(x))``.

    For ``x ≥ -1`` the direct ``log(ndtr(x))`` is fine — ``Φ`` is bounded
    below by ``≈ 0.16`` so we stay well clear of ``log(0)``.  For
    ``x < -1`` we switch to ``log(½ · erfc(-x/√2))`` which avoids the
    catastrophic cancellation that hits ``1 + erf(x/√2)`` deep in the
    left tail.
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
    """Inverse normal CDF (probit) via Beasley-Springer-Moro.

    Splits the unit interval into a central region ``[plow, phigh]``
    (``plow = 0.02425``, ``phigh = 1 - plow``) where a rational
    approximation in ``q = p − 0.5`` works, and tail regions where a
    rational approximation in ``r = √(-2·log(min(p, 1-p)))`` is used.
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
    """``x · log1p(y)`` with the convention ``0 · log1p(0) = 0`` (and 0
    propagation everywhere ``x == 0``).  Mirrors :func:`lucid.xlogy`."""
    safe_y = lucid.where(y == lucid.zeros_like(y), lucid.full_like(y, 0.0), y)
    out = x * lucid.log1p(safe_y)
    return lucid.where(x == lucid.zeros_like(x), lucid.full_like(out, 0.0), out)


def entr(x: Tensor) -> Tensor:
    """Element-wise entropy: ``-x · log(x)`` for ``x > 0``, ``0`` at
    ``x = 0``, ``NaN`` for ``x < 0`` (matches the standard reference
    framework's convention)."""
    safe_x = lucid.where(x > lucid.zeros_like(x), x, lucid.full_like(x, 1.0))
    val = -safe_x * lucid.log(safe_x)
    val = lucid.where(x == lucid.zeros_like(x), lucid.full_like(val, 0.0), val)
    val = lucid.where(x < lucid.zeros_like(x), lucid.full_like(val, float("nan")), val)
    return val


# ── Gamma family ───────────────────────────────────────────────────────────


def multigammaln(a: Tensor, p: int) -> Tensor:
    """Log of the multivariate gamma function:
    ``log Γ_p(a) = (p·(p−1)/4) · log(π) + Σ_{i=1..p} lgamma(a + (1−i)/2)``.
    """
    if int(p) < 1:
        raise ValueError(f"multigammaln requires p >= 1, got {p}")
    accum = lucid.zeros_like(a)
    for i in range(1, int(p) + 1):
        accum = accum + lucid.lgamma(a + (1.0 - float(i)) / 2.0)
    return float(p * (p - 1)) / 4.0 * math.log(math.pi) + accum


def polygamma(n: int, x: Tensor) -> Tensor:
    """``n``-th derivative of the digamma function.

    * ``n = 0`` is the digamma function — falls through to
      :func:`lucid.digamma`.
    * ``n = 1, 2, 3`` use the standard recurrence-then-asymptotic
      strategy: shift ``x`` upward by ``K=6`` via
      ``ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+1) + (−1)ⁿ⁺¹·n!/x^(n+1)``,
      accumulating the per-step corrections, then evaluate the
      Bernoulli-series asymptotic expansion at the shifted argument
      where it is well-conditioned.
    * ``n ≥ 4`` raises — extending the Bernoulli series to higher
      orders is mechanical but not currently needed.

    The reference framework only defines ``polygamma`` for non-negative
    integer ``n``; we follow that contract.
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
    """Spherical Bessel of the first kind, order 0: ``sin(x) / x`` with
    ``j₀(0) = 1``.  Related to ``lucid.sinc`` by a factor of ``π``:
    ``j₀(x) = sinc(x / π)``."""
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
    advance,
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
    """Chebyshev polynomial of the first kind ``T_n(x)``.

    Recurrence: ``T_0 = 1``, ``T_1 = x``, ``T_{k+1} = 2x·T_k - T_{k-1}``.
    """
    p0 = lucid.ones_like(x)
    p1 = x
    return _ortho_poly(x, n, p0, p1, lambda p, q, _k: 2 * x * p - q)


def chebyshev_polynomial_u(x: Tensor, n: int) -> Tensor:
    """Chebyshev polynomial of the second kind ``U_n(x)``.

    Recurrence: ``U_0 = 1``, ``U_1 = 2x``, ``U_{k+1} = 2x·U_k - U_{k-1}``.
    """
    p0 = lucid.ones_like(x)
    p1 = 2 * x
    return _ortho_poly(x, n, p0, p1, lambda p, q, _k: 2 * x * p - q)


def chebyshev_polynomial_v(x: Tensor, n: int) -> Tensor:
    """Chebyshev polynomial of the third kind ``V_n(x)``.

    Recurrence: ``V_0 = 1``, ``V_1 = 2x - 1``,
    ``V_{k+1} = 2x·V_k - V_{k-1}``.
    """
    p0 = lucid.ones_like(x)
    p1 = 2 * x - 1
    return _ortho_poly(x, n, p0, p1, lambda p, q, _k: 2 * x * p - q)


def chebyshev_polynomial_w(x: Tensor, n: int) -> Tensor:
    """Chebyshev polynomial of the fourth kind ``W_n(x)``.

    Recurrence: ``W_0 = 1``, ``W_1 = 2x + 1``,
    ``W_{k+1} = 2x·W_k - W_{k-1}``.
    """
    p0 = lucid.ones_like(x)
    p1 = 2 * x + 1
    return _ortho_poly(x, n, p0, p1, lambda p, q, _k: 2 * x * p - q)


def shifted_chebyshev_polynomial_t(x: Tensor, n: int) -> Tensor:
    """Shifted Chebyshev T (defined on ``[0, 1]`` instead of ``[-1, 1]``).

    ``T*_n(x) = T_n(2x - 1)``.
    """
    return chebyshev_polynomial_t(2 * x - 1, n)


def shifted_chebyshev_polynomial_u(x: Tensor, n: int) -> Tensor:
    """Shifted Chebyshev U: ``U*_n(x) = U_n(2x - 1)``."""
    return chebyshev_polynomial_u(2 * x - 1, n)


def shifted_chebyshev_polynomial_v(x: Tensor, n: int) -> Tensor:
    """Shifted Chebyshev V: ``V*_n(x) = V_n(2x - 1)``."""
    return chebyshev_polynomial_v(2 * x - 1, n)


def shifted_chebyshev_polynomial_w(x: Tensor, n: int) -> Tensor:
    """Shifted Chebyshev W: ``W*_n(x) = W_n(2x - 1)``."""
    return chebyshev_polynomial_w(2 * x - 1, n)


def hermite_polynomial_h(x: Tensor, n: int) -> Tensor:
    """Physicist's Hermite polynomial ``H_n(x)``.

    Recurrence: ``H_0 = 1``, ``H_1 = 2x``,
    ``H_{k+1} = 2x·H_k - 2k·H_{k-1}``.
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
    """Probabilist's Hermite polynomial ``He_n(x)``.

    Recurrence: ``He_0 = 1``, ``He_1 = x``,
    ``He_{k+1} = x·He_k - k·He_{k-1}``.
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
    """Legendre polynomial ``P_n(x)``.

    Recurrence: ``P_0 = 1``, ``P_1 = x``,
    ``(k+1)·P_{k+1} = (2k+1)·x·P_k - k·P_{k-1}``.
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
    """(Simple) Laguerre polynomial ``L_n(x)``.

    Recurrence: ``L_0 = 1``, ``L_1 = 1 - x``,
    ``(k+1)·L_{k+1} = (2k + 1 - x)·L_k - k·L_{k-1}``.
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


def _split(x: Tensor, threshold: float, small_fn, large_fn) -> Tensor:
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
    """Bessel function of the first kind, order 0.  Even function;
    accuracy ≈ 1e-7 from A&S §9.4."""
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
    """Bessel function of the first kind, order 1.  Odd function;
    accuracy ≈ 1e-7 from A&S §9.4."""
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
    """Bessel function of the second kind, order 0.  Defined for
    ``x > 0`` only.  Accuracy ≈ 1e-7 from the NR §6.5 fit."""
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
    """Bessel function of the second kind, order 1.  Defined for
    ``x > 0`` only.  Accuracy ≈ 1e-7 from the NR §6.5 fit."""
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
    """Modified Bessel function of the second kind, order 0.  Domain
    ``x > 0``; accuracy ≈ 1e-7."""
    return _split(x, 2.0, _modified_bessel_k0_small, _modified_bessel_k0_large)


def scaled_modified_bessel_k0(x: Tensor) -> Tensor:
    """``exp(x) · K0(x)`` — numerically stable form for large ``x``."""
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
    """Modified Bessel function of the second kind, order 1.  Domain
    ``x > 0``."""
    return _split(x, 2.0, _modified_bessel_k1_small, _modified_bessel_k1_large)


def scaled_modified_bessel_k1(x: Tensor) -> Tensor:
    """``exp(x) · K1(x)``."""
    return modified_bessel_k1(x) * lucid.exp(x)


# ── Hurwitz zeta ───────────────────────────────────────────────────────────


def zeta(x: Tensor, q: Tensor) -> Tensor:
    """Hurwitz zeta function ``ζ(x, q) = Σ_{k=0}^∞ (k + q)^{-x}``.

    Implementation: direct series summation with Euler–Maclaurin
    correction.  Accuracy is ≈ 1e-6 for ``x > 1`` and moderate ``q``;
    suitable for ML use cases (e.g., distribution log-densities) but
    not heroic-precision scientific computing.

    The Riemann zeta function is the special case ``ζ(x) = ζ(x, 1)``.
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
    """Alias for ``lgamma`` — log-gamma function."""
    return lucid.lgamma(x)


def psi(x: Tensor) -> Tensor:
    """Alias for ``digamma`` — digamma function ψ(x) = d/dx ln Γ(x)."""
    return lucid.digamma(x)


def expit(x: Tensor) -> Tensor:
    """Logistic sigmoid σ(x) = 1 / (1 + exp(-x)).  Alias for ``sigmoid``."""
    return lucid.sigmoid(x)


def modified_bessel_i0(x: Tensor) -> Tensor:
    """Modified Bessel function of the first kind, order 0.  Alias for ``i0``."""
    return lucid.i0(x)


def modified_bessel_i1(x: Tensor) -> Tensor:
    """Modified Bessel function of the first kind, order 1.  Alias for ``i1``."""
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
