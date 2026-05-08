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
    inv_xn: Tensor = r ** n            # 1 / xr^n.
    inv_xnp1: Tensor = inv_xn * r      # 1 / xr^(n+1).
    inv_xnp2: Tensor = inv_xnp1 * r    # 1 / xr^(n+2)  ← k=1 term.
    inv_xnp4: Tensor = inv_xnp2 * r * r  # 1 / xr^(n+4)  ← k=2 term.
    inv_xnp6: Tensor = inv_xnp4 * r * r  # 1 / xr^(n+6)  ← k=3 term.

    # Coefficients per n.  The leading two are (n−1)! and n!/2; the
    # Bernoulli triplet is B_{2k}·(2k+n−1)!/(2k)! for k=1..3.
    if n == 1:
        c_lead: float = 1.0           # 0! = 1.
        c_half: float = 0.5           # 1!/2 = 1/2.
        # k=1: B₂·2!/2! = 1/6, k=2: B₄·4!/4! = −1/30, k=3: B₆·6!/6! = 1/42.
        c1: float = 1.0 / 6.0
        c2: float = -1.0 / 30.0
        c3: float = 1.0 / 42.0
    elif n == 2:
        c_lead = 1.0                  # 1! = 1.
        c_half = 1.0                  # 2!/2 = 1.
        # k=1: B₂·3!/2! = (1/6)·3 = 1/2.
        # k=2: B₄·5!/4! = (−1/30)·5 = −1/6.
        # k=3: B₆·7!/6! = (1/42)·7 = 1/6.
        c1 = 0.5
        c2 = -1.0 / 6.0
        c3 = 1.0 / 6.0
    else:  # n == 3
        c_lead = 2.0                  # 2! = 2.
        c_half = 3.0                  # 3!/2 = 3.
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
]
