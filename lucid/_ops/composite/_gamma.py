"""The gamma family: :math:`\\ln\\Gamma` and the polygammas :math:`\\psi^{(n)}`.

``lucid.lgamma`` is :func:`_lgamma`: the Lanczos approximation (Numerical
Recipes, g = 7, n = 9; relative error below 1.5e-15 for x > 0 in float64)
on the right half, Euler's reflection on the left.

``lucid.digamma`` is ``n = 0`` and ``lucid.special.polygamma`` is every
``n``; both call :func:`_polygamma`.  They used to be two separate series
that shared one defect: neither had a left half.  The shift-by-8
recurrence only moves an argument above 8 if it started above 0, so for
``x < -8`` the asymptotic series was evaluated at a negative point and
answered ``nan``; between ``-8`` and ``0`` it answered finite numbers
that were simply wrong (``digamma(-7.9)`` was ``-3897.67`` against
``-7.54``), and the poles came back ``-inf`` where the reference has
``nan``.

The left half comes from differentiating Euler's reflection formula
:math:`\\psi(1 - x) - \\psi(x) = \\pi \\cot(\\pi x)` ``n`` times:

.. math::

    \\psi^{(n)}(x) = (-1)^n \\psi^{(n)}(1 - x) - \\pi^{n+1} P_n(\\cot \\pi x),

where :math:`\\frac{d^n}{dx^n} \\cot(\\pi x) = \\pi^n P_n(\\cot \\pi x)` and
:math:`P_n` is a polynomial with integer coefficients
(:func:`_cot_derivative_poly`).  ``1 - x`` is positive whenever ``x`` is
negative, so the right-half series covers it.
"""

import functools
import math

import lucid
from lucid._tensor.tensor import Tensor

_LANCZOS_G = 7.0
_LANCZOS_P: list[float] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
]


def _lgamma(x: Tensor) -> Tensor:
    """:math:`\\ln|\\Gamma(x)|` on the whole real line."""
    # The Lanczos series is only valid on the right half of the line, and
    # nothing here said so: ``lgamma(-0.5)`` came back NaN against the
    # reference's 1.2655, and ``lgamma(1e-30)`` came back inf against
    # 69.0776.  Γ has poles at the non-positive integers and is perfectly
    # finite everywhere between them, so the left half is not a domain
    # error — it is the half this implementation never covered.
    #
    # Euler's reflection formula supplies it:
    #
    #     lgamma(x) = log(π) − log|sin(πx)| − lgamma(1 − x)
    #
    # Both branches are evaluated, because ``where`` selects rather than
    # short-circuits, so each is given an argument clamped into its own
    # half.  Without that the losing branch divides by zero at ``z + k``
    # and the NaN reaches the *gradient* through the mask, which is the
    # usual way a where-guarded formula goes wrong.
    half = lucid.full_like(x, 0.5)
    reflect = x < half

    big = x == math.inf
    right = _lgamma_lanczos(lucid.maximum(lucid.where(big, half, x), half))
    mirror = _lgamma_lanczos(lucid.maximum(1.0 - x, half))
    # ``sin(πx)`` is zero at every integer, where Γ has its poles and the
    # answer is +inf.  Only the reflected side may see it: elsewhere it is
    # replaced by 1, so ``log`` never meets a zero it was not asked about.
    # |sin(pi x)| has period 1 and ``x - round(x)`` is exact, so the sine is
    # taken in [-pi/2, pi/2]: at pi * x the product's rounding grows with
    # |x|, and in float32 the gradient at -1000.5 came out 6.90836 against
    # digamma's 6.90875.  Only the reflected side takes it, so that +inf on
    # the other never forms inf - round(inf) = NaN under the mask.
    x_left = lucid.where(reflect, x, half)
    sin_term = lucid.abs(lucid.sin(math.pi * (x_left - lucid.round(x_left))))
    sin_safe = lucid.where(reflect, sin_term, lucid.ones_like(x))
    left = math.log(math.pi) - lucid.log(sin_safe) - mirror
    out = lucid.where(reflect, left, right)

    # The poles are named rather than computed.  ``sin(π · -1.0)`` came out
    # 1.2e-16 before the reduction above, and lgamma(-1) answered 37.78
    # instead of +inf; the reduced form is exact at integers, but -inf
    # reduces to NaN, and a pole should not hinge on rounding either way.
    at_pole = (x <= lucid.zeros_like(x)) & (x == lucid.round(x))
    out = lucid.where(at_pole, lucid.full_like(x, float("inf")), out)
    # The Lanczos form at +inf is inf - inf.  +inf is handed to x * x
    # instead, which has the same limit and, like digamma, an infinite
    # derivative there (x log x would form inf / inf in its backward).
    x_big = lucid.where(big, x, lucid.ones_like(x))
    return lucid.where(big, x_big * x_big, out)


def _lgamma_lanczos(x: Tensor) -> Tensor:
    """The Lanczos series proper — correct for ``x >= 0.5``."""
    z = x - 1.0
    t = z + (_LANCZOS_G + 0.5)
    series = lucid.full_like(x, _LANCZOS_P[0])
    for k in range(1, len(_LANCZOS_P)):
        series = series + _LANCZOS_P[k] / (z + float(k))
    return (
        0.5 * math.log(2.0 * math.pi) + (z + 0.5) * lucid.log(t) - t + lucid.log(series)
    )


# B_2, B_4, ..., B_12.  Every odd Bernoulli number past B_1 is zero, which
# is why the asymptotic series skips odd powers.
_BERNOULLI_2K: tuple[float, ...] = (
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
)

# The recurrence moves the argument up by this much before the asymptotic
# series is evaluated.  At x + 8 >= 8, six Bernoulli terms leave the n = 0
# series a truncation error near 1e-14 relative, below float64's own noise
# for the values involved.
_SHIFT = 8


@functools.cache
def _cot_derivative_poly(n: int) -> tuple[int, ...]:
    """Coefficients (lowest power first) of :math:`P_n`.

    :math:`P_0(c) = c`, and since :math:`\\frac{d}{dx} \\cot(\\pi x) =
    -\\pi (1 + \\cot^2 \\pi x)`, :math:`P_{k+1}(c) = -(1 + c^2) P_k'(c)`.
    """
    coeffs = [0, 1]
    for _ in range(n):
        deriv = [i * coeffs[i] for i in range(1, len(coeffs))]
        nxt = [0] * (len(deriv) + 2)
        for i, d in enumerate(deriv):
            nxt[i] -= d
            nxt[i + 2] -= d
        coeffs = nxt
    return tuple(coeffs)


def _positive(n: int, x: Tensor) -> Tensor:
    """:math:`\\psi^{(n)}(x)` for ``x > 0``: recurrence shift, then the series."""
    # Every power is taken of a reciprocal, never divided by: at x = +inf
    # the backward of 1 / z^m is (-m z^(m-1)) / z^(2m) = inf / inf, while
    # that of (1/z)^m is m (1/z)^(m-1) * (-1/z^2) = 0 * 0.  It also keeps a
    # float32 x = 1e-6 from passing through x^(n+1) = 1e-24 on the way to
    # 1/x^(n+1).
    xr = x + float(_SHIFT)
    r = 1.0 / xr
    r2 = r * r
    pow_r2 = lucid.ones_like(x)
    if n == 0:
        # psi(x) = psi(x + K) - sum_{k<K} 1/(x + k), and
        # psi(z) ~ log z - 1/(2z) - sum_k B_2k / (2k z^2k).
        shift = lucid.zeros_like(x)
        for k in range(_SHIFT):
            shift = shift + 1.0 / (x + float(k))
        series = lucid.log(xr) - 0.5 * r
        for k, b in enumerate(_BERNOULLI_2K, start=1):
            pow_r2 = pow_r2 * r2
            series = series - (b / (2.0 * k)) * pow_r2
        return series - shift

    # psi^(n)(x) = psi^(n)(x + K) + (-1)^(n+1) n! sum_{k<K} 1/(x + k)^(n+1), and
    # psi^(n)(z) ~ (-1)^(n+1) [ (n-1)!/z^n + n!/(2 z^(n+1))
    #                          + sum_k B_2k (2k+n-1)!/(2k)! / z^(2k+n) ].
    # The series is written with (n-1)!/z^n factored out: evaluated term by
    # term, B_2k (2k+n-1)!/(2k)! alone passes float32's range from about
    # n = 31 although the terms are tiny.  The one quantity that genuinely
    # grows like a factorial is evaluated once, in log space.
    sign = 1.0 if n % 2 == 1 else -1.0
    n_fact = float(math.factorial(n))
    shift = lucid.zeros_like(x)
    for k in range(_SHIFT):
        shift = shift + (sign * n_fact) * (1.0 / (x + float(k))) ** (n + 1)
    lead = lucid.exp(math.lgamma(n) - n * lucid.log(xr))
    bracket = 1.0 + (n / 2.0) * r
    for k, b in enumerate(_BERNOULLI_2K, start=1):
        pow_r2 = pow_r2 * r2
        # (2k+n-1)! / (n-1)! is a product of 2k integers, not a factorial.
        ratio = float(math.prod(range(n, 2 * k + n)))
        bracket = bracket + (b * ratio / float(math.factorial(2 * k))) * pow_r2
    return sign * lead * bracket + shift


def _horner(coeffs: tuple[int, ...], c: Tensor) -> Tensor:
    acc = lucid.full_like(c, float(coeffs[-1]))
    for a in reversed(coeffs[:-1]):
        acc = acc * c + float(a)
    return acc


def _polygamma(n: int, x: Tensor) -> Tensor:
    """:math:`\\psi^{(n)}(x)` on the whole real line, ``n >= 0``."""
    one = lucid.ones_like(x)
    # The poles are the non-positive integers, -inf among them.
    pole = (x <= 0.0) & (x == lucid.floor(x))
    reflect = (x < 0.0) & ~pole
    # Each element needs the right-half series at one point only — x, or
    # 1 - x where it is reflected — so the series runs once, at that point.
    # A pole gets 1 instead: ``where`` selects rather than short-circuits,
    # and a 1/0 under the mask would reach the gradient as 0 * inf.
    series = _positive(n, lucid.where(reflect, 1.0 - x, lucid.where(pole, one, x)))

    # cot(pi x) has period 1, and ``x - round(x)`` is exact, so the cotangent
    # is taken at a point r in [-1/2, 1/2] instead of at pi * x, whose
    # rounding grows with |x|.  cos(pi r) is taken as sin(pi (1/2 - |r|)),
    # whose argument is exact near |r| = 1/2: cos(pi/2) itself rounds to
    # 6e-17, and at the half-integers, where P_n(0) = 0 for even n, that
    # residue was the whole error — 4e-9 relative in polygamma(2, -1000.5).
    frac = lucid.where(reflect, x - lucid.round(x), lucid.full_like(x, 0.25))
    cot = lucid.sin(math.pi * (0.5 - lucid.abs(frac))) / lucid.sin(math.pi * frac)
    cot_term = math.pi ** (n + 1) * _horner(_cot_derivative_poly(n), cot)
    left = (series if n % 2 == 0 else -series) - cot_term
    out = lucid.where(reflect, left, series)

    if n == 0:
        # psi(x) ~ -1/x at the origin, so psi(+0) = -inf and psi(-0) = +inf;
        # the other poles have no sign and are NaN.
        at_zero = x == 0.0
        out = lucid.where(pole, lucid.full_like(x, math.nan), out)
        return lucid.where(at_zero, -1.0 / lucid.where(at_zero, x, one), out)
    # (-1)^(n+1) n! / (x + m)^(n+1): with n odd the power is even and the pole
    # is +inf from both sides; with n even the reference takes the sign
    # (-1)^(n+1) as well, and so does this.
    pole_value = math.inf if n % 2 == 1 else -math.inf
    return lucid.where(pole, lucid.full_like(x, pole_value), out)
