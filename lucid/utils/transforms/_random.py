"""Scalar random sampling for stochastic transforms.

Draws Python ``float`` / ``int`` from Lucid's global RNG, so transform
randomness honours :func:`lucid.manual_seed` (reproducible pipelines).
"""

import lucid


def rand() -> float:
    r"""Sample a uniform float in ``[0, 1)`` from Lucid's global RNG.

    Honours :func:`lucid.manual_seed` so transform stochasticity is
    reproducible across runs.  Backed by a single ``lucid.rand(1)``
    draw — the per-call overhead is one engine round-trip.

    Returns
    -------
    float
        Sampled value in ``[0, 1)``.
    """
    return float(lucid.rand(1).item())


def uniform(low: float, high: float) -> float:
    r"""Sample a uniform float in ``[low, high)`` from Lucid's RNG.

    Computed as ``low + (high - low) * rand()``; honours
    :func:`lucid.manual_seed`.  If ``high <= low`` the range is empty
    and the return value collapses toward ``low``.

    Parameters
    ----------
    low : float
        Inclusive lower bound of the sampling interval.
    high : float
        Exclusive upper bound of the sampling interval.

    Returns
    -------
    float
        Sampled value in ``[low, high)``.
    """
    return low + (high - low) * rand()


def randint(low: float, high: float) -> int:
    r"""Sample a uniform integer in ``[low, high)`` from Lucid's RNG.

    Computed by drawing a uniform float in ``[0, 1)`` and scaling by
    ``int(high) - int(low)``.  When the integer span is non-positive
    the function short-circuits to ``int(low)``.

    Parameters
    ----------
    low : float
        Inclusive lower bound (truncated to ``int``).
    high : float
        Exclusive upper bound (truncated to ``int``).

    Returns
    -------
    int
        Sampled value in ``[int(low), int(high))``, or ``int(low)``
        when the range is empty.
    """
    span = int(high) - int(low)
    if span <= 0:
        return int(low)
    return int(low) + int(rand() * span)
