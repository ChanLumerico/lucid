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


def _ramp(M: int, *, dtype: DTypeLike, device: DeviceLike) -> Tensor:
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


def _trim(window: Tensor, M: int, sym: bool) -> Tensor:
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
) -> Tensor:
    r"""Bartlett (triangular) window.

    Generates a triangular taper that ramps linearly from 0 to 1 and
    back to 0 across the window length.  Equivalent to the convolution
    of two boxcar windows, the Bartlett window provides the simplest
    non-trivial reduction of spectral leakage at the price of a wider
    main lobe.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    sym : bool, optional
        If ``True`` (default) generate a symmetric window suitable for
        filter design.  If ``False`` generate a periodic window
        suitable for use with the DFT (the length-:math:`M+1` symmetric
        form with the last sample dropped).
    dtype : DTypeLike, optional
        Desired dtype of the output tensor; defaults to ``float32``.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M`` containing the window samples.

    Notes
    -----
    Sample formula:

    .. math::

        w[n] = 1 - \left| \frac{2n}{N - 1} - 1 \right|, \qquad
        0 \le n < N,

    where :math:`N = M` (symmetric) or :math:`N = M + 1` (periodic).
    The main-lobe width is roughly :math:`8\pi / N` (about twice that
    of the rectangular window) and the peak side-lobe attenuation is
    :math:`\approx -26.5\, \text{dB}`.

    Examples
    --------
    >>> from lucid.signal.windows import bartlett
    >>> bartlett(5)
    Tensor([0.0000, 0.5000, 1.0000, 0.5000, 0.0000])
    """
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
) -> Tensor:
    r"""Cosine (half-sine) window.

    Produces a single arch of a sine curve over the window length,
    sometimes called the "sine" or "half-cosine" window.  It is a
    smoother taper than the triangular window with comparable main-lobe
    width but significantly lower side lobes.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M`` containing the window samples.

    Notes
    -----
    Sample formula:

    .. math::

        w[n] = \sin\!\left( \frac{\pi (n + 1/2)}{N} \right),
        \qquad 0 \le n < N.

    Peak side-lobe attenuation is :math:`\approx -23\, \text{dB}` and
    the main-lobe width is :math:`4\pi/N`.  Sometimes used as the
    square root of the Hann window for analysis/synthesis pairs.

    Examples
    --------
    >>> from lucid.signal.windows import cosine
    >>> cosine(4)
    Tensor([0.3827, 0.9239, 0.9239, 0.3827])
    """
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
) -> Tensor:
    r"""Hann (raised-cosine) window.

    The Hann window is a single full cycle of a raised cosine.  Its
    side-lobes roll off at :math:`-18\,\text{dB/octave}`, the fastest
    asymptotic decay among the simple cosine-sum windows, which makes
    it a popular default for STFT analysis when temporal localisation
    matters more than peak side-lobe suppression.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M`` containing the window samples.

    Notes
    -----
    Sample formula (special case of :func:`general_hamming` with
    :math:`\alpha = 0.5`):

    .. math::

        w[n] = \tfrac{1}{2}\left(
                 1 - \cos\!\left(\frac{2\pi n}{N - 1}\right)\right),
        \qquad 0 \le n < N.

    Main-lobe width :math:`8\pi/N`, peak side-lobe attenuation
    :math:`-31.5\,\text{dB}`.  Vanishes at both endpoints.

    Examples
    --------
    >>> from lucid.signal.windows import hann
    >>> hann(5)
    Tensor([0.0000, 0.5000, 1.0000, 0.5000, 0.0000])
    """
    return general_hamming(M, alpha=0.5, sym=sym, dtype=dtype, device=device)


def hamming(
    M: int,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    r"""Hamming window.

    The Hamming window is a raised cosine on a nonzero pedestal, chosen
    so that the first side-lobe is suppressed about :math:`-43\,
    \text{dB}` — substantially better than the Hann window at the cost
    of slower asymptotic side-lobe decay (only :math:`-6\,
    \text{dB/octave}`).

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M`` containing the window samples.

    Notes
    -----
    Special case of :func:`general_hamming` with :math:`\alpha = 0.54`:

    .. math::

        w[n] = 0.54 - 0.46 \cos\!\left(\frac{2\pi n}{N - 1}\right),
        \qquad 0 \le n < N.

    Main-lobe width :math:`8\pi/N`.  Endpoint values are ``0.08``
    rather than ``0``.

    Examples
    --------
    >>> from lucid.signal.windows import hamming
    >>> hamming(5)
    Tensor([0.0800, 0.5400, 1.0000, 0.5400, 0.0800])
    """
    return general_hamming(M, alpha=0.54, sym=sym, dtype=dtype, device=device)


def general_hamming(
    M: int,
    alpha: float,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    r"""Generalised Hamming-family window.

    Parametric two-term cosine window covering the Hann
    (:math:`\alpha = 0.5`) / Hamming (:math:`\alpha = 0.54`) family.
    The free parameter :math:`\alpha` controls the trade-off between
    the level of the first side-lobe and the asymptotic side-lobe
    roll-off rate.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    alpha : float
        Mixing coefficient; :math:`\alpha = 0.5` gives Hann,
        :math:`\alpha = 0.54` gives Hamming.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M`` containing the window samples.

    Notes
    -----
    Sample formula:

    .. math::

        w[n] = \alpha - (1 - \alpha)\,
               \cos\!\left( \frac{2\pi n}{N - 1} \right),
        \qquad 0 \le n < N.

    With :math:`\alpha = 25/46 \approx 0.5435` the first side-lobe is
    minimised in absolute level (the so-called "optimal" Hamming).

    Examples
    --------
    >>> from lucid.signal.windows import general_hamming
    >>> general_hamming(5, alpha=0.5)
    Tensor([0.0000, 0.5000, 1.0000, 0.5000, 0.0000])
    """
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
) -> Tensor:
    r"""Generic weighted sum of cosines.

    Builds a window of the form
    :math:`\sum_k a_k (-1)^k \cos(2\pi k n / (N - 1))`, the parent
    family from which Hann, Hamming, Blackman, Nuttall, and the
    flat-top window all derive.  Allows custom coefficient designs for
    bespoke side-lobe characteristics.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    a : list of float or tuple of float
        Coefficients :math:`a_0, a_1, \ldots`.  The :math:`k`-th
        coefficient multiplies :math:`(-1)^k \cos(2\pi k n / (N - 1))`.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M``.

    Notes
    -----
    Sample formula:

    .. math::

        w[n] = \sum_{k=0}^{K-1} (-1)^k a_k
               \cos\!\left( \frac{2\pi k n}{N - 1} \right),
        \qquad 0 \le n < N.

    Common special cases:

    * Hann: ``[0.5, 0.5]``.
    * Hamming: ``[0.54, 0.46]``.
    * Blackman: ``[0.42, 0.50, 0.08]``.
    * Nuttall: ``[0.3635819, 0.4891775, 0.1365995, 0.0106411]``.

    Examples
    --------
    >>> from lucid.signal.windows import general_cosine
    >>> general_cosine(5, [0.5, 0.5])
    Tensor([0.0000, 0.5000, 1.0000, 0.5000, 0.0000])
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
) -> Tensor:
    r"""Blackman window.

    Three-term cosine-sum window designed by Blackman and Tukey to
    drive the first three side-lobes substantially below the Hann
    level.  Peak side-lobe attenuation is :math:`\approx -58\,
    \text{dB}` with a main-lobe roughly twice as wide as the Hann
    window.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M``.

    Notes
    -----
    Sample formula (special case of :func:`general_cosine` with
    coefficients ``[0.42, 0.50, 0.08]``):

    .. math::

        w[n] = 0.42 - 0.50 \cos\!\left(\frac{2\pi n}{N - 1}\right)
                + 0.08 \cos\!\left(\frac{4\pi n}{N - 1}\right),
        \qquad 0 \le n < N.

    The slightly different "exact Blackman" uses coefficients derived
    from the first two zeros of the side-lobe envelope; the standard
    ``[0.42, 0.50, 0.08]`` truncation used here is the textbook form.

    Examples
    --------
    >>> from lucid.signal.windows import blackman
    >>> blackman(5)
    Tensor([0.0000, 0.3400, 1.0000, 0.3400, 0.0000])
    """
    return general_cosine(M, [0.42, 0.50, 0.08], sym=sym, dtype=dtype, device=device)


def nuttall(
    M: int,
    *,
    sym: bool = True,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    r"""Nuttall four-term window.

    A four-term cosine-sum window with coefficients optimised by
    Nuttall to drive the peak side-lobe to approximately :math:`-98\,
    \text{dB}`, the deepest practical suppression among four-term
    windows.  Useful for very-high-dynamic-range spectral analysis at
    the cost of a wider main lobe.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M``.

    Notes
    -----
    Sample formula (four-term :func:`general_cosine` with coefficients
    ``[0.3635819, 0.4891775, 0.1365995, 0.0106411]``):

    .. math::

        w[n] = 0.3635819
             - 0.4891775 \cos\!\left(\frac{2\pi n}{N - 1}\right)
             + 0.1365995 \cos\!\left(\frac{4\pi n}{N - 1}\right)
             - 0.0106411 \cos\!\left(\frac{6\pi n}{N - 1}\right).

    The main-lobe is roughly :math:`16\pi/N` wide.

    Examples
    --------
    >>> from lucid.signal.windows import nuttall
    >>> nuttall(5)
    Tensor([0.0004, 0.2270, 1.0000, 0.2270, 0.0004])
    """
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
) -> Tensor:
    r"""Gaussian window.

    Produces a Gaussian-shaped taper centred on the window midpoint.
    Because the Fourier transform of a Gaussian is itself a Gaussian,
    this window achieves the (continuous-time) time-frequency
    uncertainty bound and is the basis of the Gabor / short-time
    Fourier transform.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    std : float, optional
        Standard deviation in samples; defaults to ``7.0``.  Larger
        ``std`` → broader (closer to rectangular) window; smaller →
        more concentrated taper.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M``.

    Notes
    -----
    Sample formula:

    .. math::

        w[n] = \exp\!\left(
                 -\frac{1}{2}
                  \left(\frac{n - (N-1)/2}{\sigma}\right)^2
               \right), \qquad 0 \le n < N.

    Truncating an infinite Gaussian to length ``M`` produces shallow
    discontinuities at the ends; choose ``std`` small enough that the
    window has decayed substantially by the boundary.

    Examples
    --------
    >>> from lucid.signal.windows import gaussian
    >>> gaussian(5, std=1.0)
    Tensor([0.1353, 0.6065, 1.0000, 0.6065, 0.1353])
    """
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
) -> Tensor:
    r"""Generalised Gaussian window.

    Family parametrised by a shape exponent ``p`` and width ``sig``,
    interpolating between the standard Gaussian (``p = 1``) and an
    increasingly flat-topped, sharper-edged window as ``p`` grows
    large.  Useful when a Gaussian-like taper is desired but with more
    energy concentration in the centre samples.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    p : float, optional
        Shape exponent; ``p = 1`` is the standard Gaussian, ``p > 1``
        flattens the top.  Defaults to ``1.0``.
    sig : float, optional
        Width parameter analogous to ``std``; defaults to ``7.0``.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M``.

    Notes
    -----
    Sample formula:

    .. math::

        w[n] = \exp\!\left(
                 -\frac{1}{2}
                  \frac{|n - (N-1)/2|^{2p}}{\sigma^{2p}}
               \right), \qquad 0 \le n < N.

    As :math:`p \to \infty` the window approaches a rectangular shape
    of half-width :math:`\sigma`.

    Examples
    --------
    >>> from lucid.signal.windows import general_gaussian
    >>> general_gaussian(5, p=2.0, sig=1.5)
    Tensor([0.4111, 0.9023, 1.0000, 0.9023, 0.4111])
    """
    N = _length_for_sym(M, sym)
    if N <= 0:
        return lucid.zeros(0, dtype=dtype, device=device)
    n = _ramp(N, dtype=dtype, device=device)
    centred = lucid.abs(n - 0.5 * float(N - 1))
    twop = 2.0 * float(p)
    w = lucid.exp(-0.5 * (centred**twop) / (float(sig) ** twop))
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
) -> Tensor:
    r"""Exponential (Poisson) window.

    Two-sided exponential decay centred at a user-controllable point.
    Useful for emphasising one end of a frame (e.g., the most recent
    sample in an online buffer when ``center = M - 1``), or as the
    impulse-response window of an exponentially-decaying filter.

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    center : float or None, optional
        Centre of the decay.  If ``None`` (default), uses
        :math:`(N - 1) / 2` — only allowed when ``sym=True``.
    tau : float, optional
        Decay time constant in samples; defaults to ``1.0``.  Larger
        ``tau`` → slower decay, broader window.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
        ``sym=False`` requires an explicit ``center``.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M``.

    Notes
    -----
    Sample formula:

    .. math::

        w[n] = \exp\!\left( -\frac{|n - n_c|}{\tau} \right),
        \qquad 0 \le n < N,

    where :math:`n_c` is ``center``.  The double-sided exponential
    corresponds to the impulse response of a symmetric one-pole IIR
    filter pair.

    Examples
    --------
    >>> from lucid.signal.windows import exponential
    >>> exponential(5, tau=2.0)
    Tensor([0.3679, 0.6065, 1.0000, 0.6065, 0.3679])
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
) -> Tensor:
    r"""Kaiser–Bessel window.

    Near-optimal window in the sense that it provides the largest
    fraction of energy in the main lobe for a given main-lobe width.
    A single parameter :math:`\beta` smoothly traverses the trade-off
    between time and frequency resolution; popular choices include
    :math:`\beta \approx 5` (≈ −37 dB side-lobes), :math:`\beta \approx
    8.6` (≈ −90 dB), and :math:`\beta \approx 12` (≈ −124 dB).

    Parameters
    ----------
    M : int
        Number of samples in the output window; must be ``>= 0``.
    beta : float, optional
        Shape parameter; defaults to ``12.0``.  Larger ``beta`` → wider
        main lobe, lower side lobes.  ``beta = 0`` yields the
        rectangular window; ``beta ≈ 5`` approximates the Hamming
        window.
    sym : bool, optional
        Symmetric (``True``, default) or periodic (``False``) variant.
    dtype : DTypeLike, optional
        Desired dtype of the output tensor.
    device : DeviceLike, optional
        Target device for the output tensor.

    Returns
    -------
    Tensor
        1-D tensor of length ``M``.

    Notes
    -----
    Sample formula:

    .. math::

        w[n] = \frac{I_0\!\left(\beta\,
                  \sqrt{1 - \left(\frac{2n}{N-1} - 1\right)^2}\right)}
                  {I_0(\beta)}, \qquad 0 \le n < N,

    where :math:`I_0` is the modified Bessel function of the first
    kind, order 0.  The Kaiser-Bessel window is the discrete-time
    approximation to the continuous prolate-spheroidal window; the
    side-lobe attenuation in dB is approximately :math:`8.7\,\beta` for
    moderate :math:`\beta`.

    Examples
    --------
    >>> from lucid.signal.windows import kaiser
    >>> kaiser(5, beta=8.0)
    Tensor([0.0046, 0.3464, 1.0000, 0.3464, 0.0046])
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
            + 3.0899424 * t**2
            + 1.2067492 * t**3
            + 0.2659732 * t**4
            + 0.0360768 * t**5
            + 0.0045813 * t**6
        )
    y = 3.75 / ax
    poly = (
        0.39894228
        + 0.01328592 * y
        + 0.00225319 * y**2
        - 0.00157565 * y**3
        + 0.00916281 * y**4
        - 0.02057706 * y**5
        + 0.02635537 * y**6
        - 0.01647633 * y**7
        + 0.00392377 * y**8
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
