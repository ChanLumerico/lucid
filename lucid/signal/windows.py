"""Window functions for spectral analysis (``lucid.signal.windows``).

All twelve windows are pure-Python composites built on ``lucid.arange`` and
elementwise math — no numpy, no engine work.  Each window is parameterised
by ``M`` (the window length, must be ``>= 0``) plus a few shape-specific
hyperparameters.

Symmetry / periodic flag
------------------------
Every window accepts a keyword-only ``sym: bool = True`` flag.  The
standard reference convention is:

* ``sym=True``   — generate a symmetric window of length ``M`` (use this
  for filter design, where the window's symmetry matters).
* ``sym=False``  — generate a *periodic* window of length ``M`` (use this
  for spectral analysis / DFT — the periodic form has length ``M+1`` then
  drops the last sample, so adjacent windows tile cleanly).

The two forms differ only at boundary samples; internally we compute the
length-``M+1`` form when ``sym=False`` and slice off the last point.
"""

import math
from typing import TYPE_CHECKING

import lucid
from lucid._types import DeviceLike, DTypeLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── shared helpers ─────────────────────────────────────────────────────────


def _ramp(M: int, *, dtype: DTypeLike, device: DeviceLike) -> "Tensor":
    """``arange(M)`` cast to a F32 (default) ramp.

    Pulled out as a helper because every window starts from this index
    vector and threads ``dtype`` / ``device`` through identically.
    """
    return lucid.arange(0.0, float(M), 1.0, dtype=dtype, device=device)


def _length_for_sym(M: int, sym: bool) -> int:
    """When ``sym=False`` we compute the symmetric form for ``M+1`` samples
    and return the first ``M``.  Returning the working length here keeps
    every window's body uniform."""
    if M < 0:
        raise ValueError(f"window length must be >= 0, got {M}")
    return M if sym else M + 1


def _trim(window: "Tensor", M: int, sym: bool) -> "Tensor":
    """Drop the last sample when ``sym=False`` so the periodic form has the
    user-requested length ``M``."""
    return window if sym else window[:M]


# ── one-parameter shape windows ────────────────────────────────────────────


def bartlett(
    M: int,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """Triangular window: ``w[n] = 1 - |2n/(N-1) - 1|``."""
    N = _length_for_sym(M, sym)
    if N <= 1:
        return lucid.ones(M if M > 0 else 0, dtype=dtype, device=device)
    n = _ramp(N, dtype=dtype, device=device)
    w = 1.0 - lucid.abs(2.0 * n / float(N - 1) - 1.0)
    return _trim(w, M, sym)


def cosine(
    M: int,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """Cosine (a.k.a. half-sine) window: ``w[n] = sin(pi * (n + 0.5) / N)``."""
    N = _length_for_sym(M, sym)
    if N <= 0:
        return lucid.zeros(0, dtype=dtype, device=device)
    n = _ramp(N, dtype=dtype, device=device)
    w = lucid.sin(math.pi * (n + 0.5) / float(N))
    return _trim(w, M, sym)


def hann(
    M: int,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """Hann window — special case of ``general_hamming(alpha=0.5)``."""
    return general_hamming(M, alpha=0.5, sym=sym, dtype=dtype, device=device)


def hamming(
    M: int,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """Hamming window — special case of ``general_hamming(alpha=0.54)``."""
    return general_hamming(M, alpha=0.54, sym=sym, dtype=dtype, device=device)


def general_hamming(
    M: int,
    alpha: float,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """``alpha - (1 - alpha) * cos(2*pi*n / (N-1))`` — Hann is α=0.5,
    Hamming is α=0.54."""
    N = _length_for_sym(M, sym)
    if N <= 1:
        return lucid.ones(M if M > 0 else 0, dtype=dtype, device=device)
    n = _ramp(N, dtype=dtype, device=device)
    w = alpha - (1.0 - alpha) * lucid.cos(2.0 * math.pi * n / float(N - 1))
    return _trim(w, M, sym)


# ── multi-term cosine windows ──────────────────────────────────────────────


def general_cosine(
    M: int,
    a: list[float] | tuple[float, ...],
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """Weighted sum of cosines:
    ``sum_k a[k] * (-1)^k * cos(2*pi*k*n/(N-1))``.

    Reduces to Hann (a=[0.5, 0.5]), Hamming (a=[0.54, 0.46]), Blackman,
    Nuttall, etc., depending on the coefficients.
    """
    N = _length_for_sym(M, sym)
    if N <= 1:
        return lucid.ones(M if M > 0 else 0, dtype=dtype, device=device)
    n = _ramp(N, dtype=dtype, device=device)
    base = 2.0 * math.pi * n / float(N - 1)
    w = lucid.full_like(n, float(a[0]))
    sign = -1.0
    for k in range(1, len(a)):
        w = w + sign * float(a[k]) * lucid.cos(float(k) * base)
        sign = -sign
    return _trim(w, M, sym)


def blackman(
    M: int,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """3-term Blackman window: ``a = [0.42, 0.50, 0.08]``."""
    return general_cosine(M, [0.42, 0.50, 0.08], sym=sym, dtype=dtype, device=device)


def nuttall(
    M: int,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """4-term Blackman-Nuttall window."""
    return general_cosine(
        M,
        [0.3635819, 0.4891775, 0.1365995, 0.0106411],
        sym=sym,
        dtype=dtype,
        device=device,
    )


# ── Gaussian-family windows ────────────────────────────────────────────────


def gaussian(
    M: int,
    std: float = 7.0,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """Gaussian window: ``exp(-0.5 * ((n - (N-1)/2) / std)^2)``."""
    N = _length_for_sym(M, sym)
    if N <= 0:
        return lucid.zeros(0, dtype=dtype, device=device)
    n = _ramp(N, dtype=dtype, device=device)
    centred = n - 0.5 * float(N - 1)
    w = lucid.exp(-0.5 * (centred / float(std)) ** 2)
    return _trim(w, M, sym)


def general_gaussian(
    M: int,
    p: float = 1.0,
    sig: float = 7.0,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """Generalised Gaussian: ``exp(-0.5 * |n - (N-1)/2| ** (2*p) / sig ** (2*p))``.

    ``p=1`` reduces to the standard Gaussian; larger ``p`` flattens the top.
    """
    N = _length_for_sym(M, sym)
    if N <= 0:
        return lucid.zeros(0, dtype=dtype, device=device)
    n = _ramp(N, dtype=dtype, device=device)
    centred = lucid.abs(n - 0.5 * float(N - 1))
    twop = 2.0 * float(p)
    w = lucid.exp(-0.5 * (centred ** twop) / (float(sig) ** twop))
    return _trim(w, M, sym)


# ── exponential window ────────────────────────────────────────────────────


def exponential(
    M: int,
    center: float | None = None,
    tau: float = 1.0,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """Exponential decay around ``center``: ``exp(-|n - center| / tau)``.

    ``center=None`` defaults to ``(N-1)/2`` for a symmetric window.
    """
    N = _length_for_sym(M, sym)
    if N <= 0:
        return lucid.zeros(0, dtype=dtype, device=device)
    if center is None:
        if not sym:
            raise ValueError(
                "exponential: center=None requires sym=True (the periodic "
                "form has no canonical default centre)."
            )
        center = 0.5 * float(N - 1)
    n = _ramp(N, dtype=dtype, device=device)
    w = lucid.exp(-lucid.abs(n - float(center)) / float(tau))
    return _trim(w, M, sym)


# ── Kaiser window — uses the modified Bessel function I0 ───────────────────


def kaiser(
    M: int,
    beta: float = 12.0,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> "Tensor":
    """Kaiser-Bessel window: ``I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta)``.

    ``beta`` controls the main-lobe / side-lobe trade-off (larger beta →
    wider main lobe, lower side lobes).  Implemented over ``lucid.i0``.
    """
    N = _length_for_sym(M, sym)
    if N <= 1:
        return lucid.ones(M if M > 0 else 0, dtype=dtype, device=device)
    n = _ramp(N, dtype=dtype, device=device)
    arg = 2.0 * n / float(N - 1) - 1.0
    radical = lucid.sqrt(lucid.clip(1.0 - arg * arg, 0.0, 1.0))
    num = lucid.i0(float(beta) * radical)
    denom = float(_i0_scalar(float(beta)))
    w = num / denom
    return _trim(w, M, sym)


def _i0_scalar(x: float) -> float:
    """Scalar ``I0(x)`` using the same Abramowitz polynomial Lucid uses
    in :func:`lucid.i0`.  Used to compute the Kaiser denominator without
    allocating a length-1 tensor."""
    ax = abs(x)
    if ax <= 3.75:
        t = (x / 3.75) ** 2
        return (
            1.0
            + 3.5156229 * t
            + 3.0899424 * t ** 2
            + 1.2067492 * t ** 3
            + 0.2659732 * t ** 4
            + 0.0360768 * t ** 5
            + 0.0045813 * t ** 6
        )
    y = 3.75 / ax
    poly = (
        0.39894228
        + 0.01328592 * y
        + 0.00225319 * y ** 2
        - 0.00157565 * y ** 3
        + 0.00916281 * y ** 4
        - 0.02057706 * y ** 5
        + 0.02635537 * y ** 6
        - 0.01647633 * y ** 7
        + 0.00392377 * y ** 8
    )
    return poly * math.exp(ax) / math.sqrt(ax)


__all__ = [
    "bartlett",
    "blackman",
    "cosine",
    "exponential",
    "gaussian",
    "general_cosine",
    "general_hamming",
    "general_gaussian",
    "hamming",
    "hann",
    "kaiser",
    "nuttall",
]
