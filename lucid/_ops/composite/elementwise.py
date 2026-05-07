"""Element-wise composite ops layered on the engine.

Three subgroups:

1. **Aliases** for engine ops PyTorch exposes under multiple names
   (``absolute`` ↔ ``abs``, ``subtract`` ↔ ``sub``, etc.).
2. **Inverse-hyperbolic** functions composed from ``log`` and ``sqrt``.
3. **Specials** — ``expm1``, ``sinc``, ``heaviside``, ``xlogy``, ``logit``,
   ``signbit``, ``float_power``, ``fmax`` / ``fmin`` — each implemented as
   a small expression over engine primitives, so autograd works without
   any extra wiring.
"""

import math
import math as _math
from typing import TYPE_CHECKING

import lucid
from lucid._ops.composite._shared import _is_tensor
from lucid._types import Scalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Aliases ────────────────────────────────────────────────────────────────


def absolute(x: Tensor) -> Tensor:
    return lucid.abs(x)


def negative(x: Tensor) -> Tensor:
    return lucid.neg(x)


def positive(x: Tensor) -> Tensor:
    """PyTorch parity: returns the input unchanged."""
    return x


def subtract(a: Tensor, b: Tensor | Scalar, *, alpha: float = 1.0) -> Tensor:
    if alpha == 1.0:
        return a - b
    return a - (b * alpha)


def multiply(a: Tensor, b: Tensor | Scalar) -> Tensor:
    return a * b


def divide(a: Tensor, b: Tensor | Scalar) -> Tensor:
    return a / b


def true_divide(a: Tensor, b: Tensor | Scalar) -> Tensor:
    return a / b


def rsub(a: Tensor, b: Tensor | Scalar, *, alpha: float = 1.0) -> Tensor:
    """``b - alpha * a`` — reverse subtract, mirroring ``torch.rsub``."""
    if alpha == 1.0:
        return b - a
    return b - (a * alpha)


def arctan2(y: Tensor, x: Tensor) -> Tensor:
    return lucid.atan2(y, x)


# ── Inverse hyperbolic (composed) ──────────────────────────────────────────


def arccosh(x: Tensor) -> Tensor:
    return lucid.log(x + lucid.sqrt(x * x - 1.0))


def acosh(x: Tensor) -> Tensor:
    return arccosh(x)


def arcsinh(x: Tensor) -> Tensor:
    return lucid.log(x + lucid.sqrt(x * x + 1.0))


def asinh(x: Tensor) -> Tensor:
    return arcsinh(x)


def arctanh(x: Tensor) -> Tensor:
    return lucid.log((1.0 + x) / (1.0 - x)) * 0.5


def atanh(x: Tensor) -> Tensor:
    return arctanh(x)


# ── Specials ───────────────────────────────────────────────────────────────


def expm1(x: Tensor) -> Tensor:
    """``exp(x) - 1`` — loses some precision near 0 vs a true engine primitive."""
    return lucid.exp(x) - 1.0


def sinc(x: Tensor) -> Tensor:
    """Normalised sinc: ``sin(pi*x) / (pi*x)``, with ``sinc(0) = 1``."""
    px = x * math.pi
    is_zero = x == 0.0
    safe_px = lucid.where(is_zero, lucid.full_like(x, 1.0), px)
    val = lucid.sin(safe_px) / safe_px
    return lucid.where(is_zero, lucid.full_like(x, 1.0), val)


def heaviside(x: Tensor, values: Tensor | Scalar) -> Tensor:
    """Heaviside step: 0 for x<0, 1 for x>0, ``values`` for x==0."""
    if not _is_tensor(values):
        values = lucid.full_like(x, float(values))
    return lucid.where(
        x == 0.0,
        values,
        lucid.where(x > 0.0, lucid.full_like(x, 1.0), lucid.full_like(x, 0.0)),
    )


def xlogy(x: Tensor | Scalar, y: Tensor | Scalar) -> Tensor:
    """``x * log(y)``, with the convention 0 * log(0) = 0."""
    if not _is_tensor(x):
        x = lucid.tensor(float(x))
    if not _is_tensor(y):
        y = lucid.tensor(float(y))
    safe_y = lucid.where(y == 0.0, lucid.full_like(y, 1.0), y)
    out = x * lucid.log(safe_y)
    return lucid.where(x == 0.0, lucid.full_like(out, 0.0), out)


def logit(x: Tensor, eps: float | None = None) -> Tensor:
    """``log(x / (1-x))``, optionally clamping to ``[eps, 1-eps]`` first."""
    if eps is not None:
        x = lucid.clamp(x, eps, 1.0 - eps)
    return lucid.log(x / (1.0 - x))


def signbit(x: Tensor) -> Tensor:
    """True where ``x < 0``."""
    return x < 0.0


def float_power(x: Tensor | Scalar, y: Tensor | Scalar) -> Tensor:
    """``pow`` always done in F64 (matches ``torch.float_power``)."""
    if _is_tensor(x):
        x = x.to(dtype=lucid.float64)
    if _is_tensor(y):
        y = y.to(dtype=lucid.float64)
    if not _is_tensor(x) and _is_tensor(y):
        x = lucid.tensor(float(x), dtype=lucid.float64, device=y.device)
    if not _is_tensor(y) and _is_tensor(x):
        y = lucid.tensor(float(y), dtype=lucid.float64, device=x.device)
    return lucid.pow(x, y)


def fmax(a: Tensor, b: Tensor) -> Tensor:
    """Like ``maximum`` but returns the non-NaN value when one side is NaN."""
    a_is_nan = lucid.isnan(a)
    b_is_nan = lucid.isnan(b)
    m = lucid.maximum(a, b)
    m = lucid.where(a_is_nan, b, m)
    m = lucid.where(b_is_nan, a, m)
    return m


def fmin(a: Tensor, b: Tensor) -> Tensor:
    """Like ``minimum`` but returns the non-NaN value when one side is NaN."""
    a_is_nan = lucid.isnan(a)
    b_is_nan = lucid.isnan(b)
    m = lucid.minimum(a, b)
    m = lucid.where(a_is_nan, b, m)
    m = lucid.where(b_is_nan, a, m)
    return m


# ── Special math functions ─────────────────────────────────────────────────


def erfc(x: Tensor) -> Tensor:
    """Complementary error function: ``erfc(x) = 1 - erf(x)``."""
    return lucid.full_like(x, 1.0) - lucid.erf(x)


def copysign(x: Tensor, y: Tensor) -> Tensor:
    """Return a tensor with magnitudes from ``x`` and signs from ``y``."""
    return lucid.where(y < 0.0, -lucid.abs(x), lucid.abs(x))


def ldexp(input: Tensor, exponent: Tensor | Scalar) -> Tensor:
    """``input * 2 ** exponent`` element-wise (differentiable w.r.t. both)."""
    return input * lucid.exp(exponent * math.log(2.0))


# ── Integer math (non-differentiable) ────────────────────────────────────


def gcd(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise greatest common divisor (integer tensors)."""
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    n = int(flat_x.shape[0])
    result = [_math.gcd(int(flat_x[i].item()), int(flat_y[i].item())) for i in range(n)]
    return lucid.tensor(result, dtype=x.dtype, device=x.device).reshape(x.shape)


def lcm(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise least common multiple (integer tensors)."""
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    n = int(flat_x.shape[0])
    result = [_math.lcm(int(flat_x[i].item()), int(flat_y[i].item())) for i in range(n)]
    return lucid.tensor(result, dtype=x.dtype, device=x.device).reshape(x.shape)


# ── Log-gamma and Digamma ─────────────────────────────────────────────────
#
# Both are implemented as pure composites over engine primitives so that
# autograd flows through them automatically.
#
# lgamma: Lanczos approximation (Numerical Recipes, g=7, n=9).
#   Valid for x > 0.  Using the full Lanczos series keeps relative error
#   below 1.5e-15 for real x > 0 (float64).
#
# digamma: shift via recurrence ψ(x) = ψ(x+8) − Σ 1/(x+k) to push the
#   argument above 8, then apply the asymptotic series.

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


def lgamma(x: Tensor) -> Tensor:
    """Natural log of the gamma function via the Lanczos approximation."""
    z = x - 1.0
    t = z + (_LANCZOS_G + 0.5)
    series = lucid.full_like(x, _LANCZOS_P[0])
    for k in range(1, len(_LANCZOS_P)):
        series = series + _LANCZOS_P[k] / (z + float(k))
    return (
        0.5 * math.log(2.0 * math.pi) + (z + 0.5) * lucid.log(t) - t + lucid.log(series)
    )


def digamma(x: Tensor) -> Tensor:
    """Digamma function ψ(x) = d/dx ln Γ(x) via recurrence + asymptotic series."""
    # Shift to xr = x + 8 accumulating the correction sum.
    # ψ(x) = ψ(x+8) − 1/x − 1/(x+1) − … − 1/(x+7)
    correction = lucid.zeros_like(x)
    for k in range(8):
        correction = correction + 1.0 / (x + float(k))
    xr = x + 8.0
    # Asymptotic expansion: ψ(xr) ≈ ln(xr) − 1/(2xr) − B2/(2xr²) + B4/(4xr⁴) − B6/(6xr⁶)
    # Bernoulli-based: B2=1/6, B4=-1/30, B6=1/42
    r = 1.0 / xr
    r2 = r * r
    psi = lucid.log(xr) - 0.5 * r - r2 / 12.0 + r2 * r2 / 120.0 - r2 * r2 * r2 / 252.0
    return psi - correction


# ── Modified Bessel function I₀ ───────────────────────────────────────────
#
# Uses the Abramowitz & Stegun polynomial approximation (Table 9.8.1):
#   |x| ≤ 3.75 : series in (x/3.75)²
#   |x| > 3.75 : series in 3.75/|x|, multiplied by exp(|x|)/√|x|

_I0_SMALL_COEFFS: list[float] = [
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.0360768,
    0.0045813,
]
_I0_LARGE_COEFFS: list[float] = [
    0.39894228,
    0.01328592,
    0.00225319,
    -0.00157565,
    0.00916281,
    -0.02057706,
    0.02635537,
    -0.01647633,
    0.00392377,
]


def i0(x: Tensor) -> Tensor:
    """Modified Bessel function of the first kind, order 0."""
    ax = lucid.abs(x)
    # Guard: avoid division by zero in large-argument branch (ax == 0 → use small branch)
    ax_safe = lucid.where(ax == lucid.zeros_like(ax), lucid.full_like(ax, 1.0), ax)

    # Small-argument branch: polynomial in t = (x / 3.75)²
    t_small = (x * (1.0 / 3.75)) ** 2
    val_small = lucid.full_like(x, _I0_SMALL_COEFFS[0])
    t_pow = lucid.ones_like(x)
    for c in _I0_SMALL_COEFFS[1:]:
        t_pow = t_pow * t_small
        val_small = val_small + c * t_pow

    # Large-argument branch: polynomial in y = 3.75 / |x|
    y_large = 3.75 / ax_safe
    val_large = lucid.full_like(x, _I0_LARGE_COEFFS[0])
    y_pow = lucid.ones_like(x)
    for c in _I0_LARGE_COEFFS[1:]:
        y_pow = y_pow * y_large
        val_large = val_large + c * y_pow
    val_large = val_large * lucid.exp(ax_safe) / lucid.sqrt(ax_safe)

    return lucid.where(ax <= 3.75, val_small, val_large)


__all__ = [
    "absolute",
    "negative",
    "positive",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "rsub",
    "arctan2",
    "arccosh",
    "acosh",
    "arcsinh",
    "asinh",
    "arctanh",
    "atanh",
    "expm1",
    "sinc",
    "heaviside",
    "xlogy",
    "logit",
    "signbit",
    "float_power",
    "fmax",
    "fmin",
    "erfc",
    "copysign",
    "ldexp",
    "gcd",
    "lcm",
    "lgamma",
    "digamma",
    "i0",
]
