"""
Random-stream plumbing shared by the samplers, ``random_split`` and the
``DataLoader``.

Everything random in ``lucid.utils.data`` draws from Lucid's Philox
generators — never from Python's ``random`` module — so a single
:func:`lucid.manual_seed` reproduces a whole data pipeline, and an
explicit :class:`lucid.Generator` isolates one.
"""

import random
from typing import cast

import lucid
from lucid._factories.random import Generator, randint, randperm
from lucid._tensor.tensor import Tensor

# Seeds handed to ``Generator`` / ``manual_seed`` stay inside the signed
# 64-bit range, so ``base_seed + worker_id`` cannot overflow either.
_SEED_HIGH = (1 << 63) - 1
_UNSIGNED_64 = 1 << 64


def _as_generator(generator: object, owner: str) -> Generator | None:
    r"""Normalise a ``generator=`` argument to a :class:`lucid.Generator`.

    ``None`` stays ``None`` (draw from the default generator, which
    :func:`lucid.manual_seed` controls).  A :class:`lucid.Generator` is
    used as is and advances from epoch to epoch.  An ``int`` (or any other
    seed :class:`random.Random` accepts) is turned into one private
    generator *once*, so it reproduces the whole sequence of epochs
    rather than repeating the first epoch forever.
    """
    if generator is None or isinstance(generator, Generator):
        return generator
    if isinstance(generator, int) and not isinstance(generator, bool):
        return Generator(generator % _UNSIGNED_64)
    try:
        seed = random.Random(generator).getrandbits(63)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            f"{owner}: generator must be a lucid.Generator, an int seed, or "
            f"None — got {type(generator).__name__}."
        ) from None
    return Generator(seed)


def _draw_seed(generator: Generator | None) -> int:
    r"""Draw a fresh 63-bit seed from ``generator`` (default: Lucid's).

    ``lucid.randint`` rather than ``rand``: the integer draw carries the
    full 63 bits, where a float draw would collapse the seed space.
    """
    return int(randint(0, _SEED_HIGH, (1,), generator=generator).item())


def _uniform_doubles(count: int, generator: Generator | None) -> Tensor:
    r"""``count`` uniform ``float64`` draws in ``[0, 1)``, 53 bits each.

    Built from integer draws so that every double in the grid
    ``k / 2**53`` is reachable — an inverse-CDF lookup over a long weight
    vector needs that resolution to reach its small-probability tail.
    """
    ints = randint(0, 1 << 53, (count,), generator=generator)
    return ints.to(lucid.float64) * (1.0 / (1 << 53))


def _permutation(n: int, generator: Generator | None) -> list[int]:
    r"""A uniform random permutation of ``range(n)`` as a Python list."""
    if n == 0:
        return []
    return cast(list[int], randperm(n, generator=generator).tolist())
