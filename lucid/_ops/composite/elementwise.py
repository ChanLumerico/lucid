"""Element-wise composite ops layered on the engine.

Three subgroups:

1. **Aliases** for engine ops the reference framework exposes under multiple names
   (``absolute`` ↔ ``abs``, ``subtract`` ↔ ``sub``, etc.).
2. **Inverse-hyperbolic** functions composed from ``log`` and ``sqrt``.
3. **Specials** — ``expm1``, ``sinc``, ``heaviside``, ``xlogy``, ``logit``,
   ``signbit``, ``float_power``, ``fmax`` / ``fmin`` — each implemented as
   a small expression over engine primitives, so autograd works without
   any extra wiring.
"""

import math
import math as _math
from typing import TYPE_CHECKING, cast

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
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
    """Reference-framework parity: returns the input unchanged."""
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
    """``b - alpha * a`` — reverse subtract, the reference framework's ``rsub``."""
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
        values = lucid.full_like(x, float(cast(float, values)))
    return lucid.where(
        x == 0.0,
        values,
        lucid.where(x > 0.0, lucid.full_like(x, 1.0), lucid.full_like(x, 0.0)),
    )


def xlogy(x: Tensor | Scalar, y: Tensor | Scalar) -> Tensor:
    """``x * log(y)``, with the convention 0 * log(0) = 0."""
    if not _is_tensor(x):
        x = lucid.tensor(float(cast(float, x)))
    if not _is_tensor(y):
        y = lucid.tensor(float(cast(float, y)))
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
    """``pow`` always done in F64 (matches the reference framework's ``float_power``)."""
    if _is_tensor(x):
        x = x.to(dtype=lucid.float64)
    if _is_tensor(y):
        y = y.to(dtype=lucid.float64)
    if not _is_tensor(x) and _is_tensor(y):
        x = lucid.tensor(float(cast(float, x)), dtype=lucid.float64, device=y.device)
    if not _is_tensor(y) and _is_tensor(x):
        y = lucid.tensor(float(cast(float, y)), dtype=lucid.float64, device=x.device)
    return lucid.pow(cast(Tensor, x), cast(Tensor, y))


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
    exp_t: Tensor = (
        exponent
        if _is_tensor(exponent)
        else lucid.tensor(cast(float, exponent), dtype=input.dtype, device=input.device)
    )
    return input * lucid.exp(exp_t * math.log(2.0))


def frexp(input: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose ``input`` into mantissa ``m`` and exponent ``e`` such that
    ``input = m * 2**e`` with ``|m|`` in ``[0.5, 1)`` (or ``m == 0``).

    Returns a ``(mantissa, exponent)`` tuple.  Exponent dtype is int32.
    Non-finite inputs propagate: NaN → (NaN, 0); ±inf → (±inf, 0); 0 → (0, 0).
    """
    abs_x = lucid.abs(input)
    is_zero = abs_x == 0.0
    is_nonfinite = lucid.logical_or(lucid.isinf(input), lucid.isnan(input))
    safe_abs = lucid.where(
        lucid.logical_or(is_zero, is_nonfinite),
        lucid.full_like(input, 1.0),
        abs_x,
    )
    # e = floor(log2(|x|)) + 1 puts |m| in [0.5, 1).
    e_float = lucid.floor(lucid.log2(safe_abs)) + 1.0
    e_float = lucid.where(
        lucid.logical_or(is_zero, is_nonfinite),
        lucid.full_like(e_float, 0.0),
        e_float,
    )
    m = input * lucid.exp(-e_float * math.log(2.0))
    m = lucid.where(is_zero, lucid.full_like(m, 0.0), m)
    m = lucid.where(is_nonfinite, input, m)
    e_i32 = e_float.to(dtype=lucid.int32)
    return m, e_i32


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


def softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Softmax along ``dim`` (default: last axis)."""
    axis = dim if dim is not None else -1
    return _wrap(_C_engine.softmax(_unwrap(x), axis))


def log_softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Log-softmax along ``dim`` (default: last axis)."""
    axis = dim if dim is not None else -1
    sm = _C_engine.softmax(_unwrap(x), axis)
    return _wrap(_C_engine.log(sm))


def floor_divide(a: Tensor, b: Tensor | Scalar) -> Tensor:
    """Element-wise floor division: ``floor(a / b)``."""
    return (a / b).floor()


def diag_embed(
    x: Tensor,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> Tensor:
    """Embed the last dimension of ``x`` as the diagonal of a new matrix.

    For a 1-D input of length ``n`` returns shape ``(n+|offset|, n+|offset|)``.
    For batch inputs the last dimension is embedded; ``dim1``/``dim2`` select
    which two axes of the *output* carry the matrix (default: last two).
    """
    n = int(x.shape[-1])
    size = n + abs(offset)
    batch_shape = list(x.shape[:-1])
    row_off = max(0, -offset)
    col_off = max(0, offset)

    # Build (size, size) diagonal mask by padding eye(n) with zero rows/cols.
    eye_n = lucid.eye(n, dtype=x.dtype, device=x.device)
    if col_off > 0:
        eye_n = lucid.cat(
            [lucid.zeros((n, col_off), dtype=x.dtype, device=x.device), eye_n], dim=1
        )
    right = size - n - col_off
    if right > 0:
        eye_n = lucid.cat(
            [eye_n, lucid.zeros((n, right), dtype=x.dtype, device=x.device)], dim=1
        )
    if row_off > 0:
        eye_n = lucid.cat(
            [lucid.zeros((row_off, size), dtype=x.dtype, device=x.device), eye_n], dim=0
        )
    bot = size - n - row_off
    if bot > 0:
        eye_n = lucid.cat(
            [eye_n, lucid.zeros((bot, size), dtype=x.dtype, device=x.device)], dim=0
        )
    # eye_n is now (size, size) with the diagonal at (row_off+i, col_off+i).

    # Broadcast x (..., n) against diagonal mask (size, size).
    # For offset==0: n==size, so x.reshape(..., n, 1) * eye_n.reshape(1,...,n,n).
    # For offset!=0: pad x to (..., size) by prepending/appending zeros.
    if offset == 0:
        diag_view = eye_n.reshape([1] * len(batch_shape) + [size, size])
        return x.reshape(batch_shape + [n, 1]) * diag_view
    else:
        # Expand x to (..., size) with zeros at padded positions.
        pad_before = lucid.zeros(
            tuple(batch_shape + [row_off]), dtype=x.dtype, device=x.device
        )
        pad_after = lucid.zeros(
            tuple(batch_shape + [size - n - row_off]), dtype=x.dtype, device=x.device
        )
        x_pad = lucid.cat([pad_before, x, pad_after], dim=-1)  # (..., size)
        diag_view = eye_n.reshape([1] * len(batch_shape) + [size, size])
        return x_pad.reshape(batch_shape + [size, 1]) * diag_view


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
    "frexp",
    "gcd",
    "lcm",
    "lgamma",
    "digamma",
    "i0",
    "softmax",
    "log_softmax",
    "floor_divide",
    "diag_embed",
]
