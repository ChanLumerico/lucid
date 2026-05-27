"""Scalar random sampling for stochastic transforms.

Draws Python ``float`` / ``int`` from Lucid's global RNG, so transform
randomness honours :func:`lucid.manual_seed` (reproducible pipelines).
"""

import lucid


def rand() -> float:
    """A uniform float in ``[0, 1)`` from Lucid's RNG."""
    return float(lucid.rand(1).item())


def uniform(low: float, high: float) -> float:
    """A uniform float in ``[low, high)``."""
    return low + (high - low) * rand()


def randint(low: float, high: float) -> int:
    """A uniform integer in ``[low, high)`` (empty range → ``low``)."""
    span = int(high) - int(low)
    if span <= 0:
        return int(low)
    return int(low) + int(rand() * span)
