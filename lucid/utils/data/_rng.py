"""
Random-stream plumbing shared by the samplers, ``random_split`` and the
``DataLoader``.

Everything random in ``lucid.utils.data`` draws from Lucid's Philox
generators — never from Python's ``random`` module — so a single
:func:`lucid.manual_seed` reproduces a whole data pipeline, and an
explicit :class:`lucid.Generator` isolates one.

Every draw here is made on the host (``device="cpu"``), whatever
:func:`lucid.set_default_device` says.  Indices, seeds and sampling
weights are host data that end up as Python ints; drawing them on the
default device would make a shuffle depend on where the *model* lives,
and Metal holds neither ``float64`` nor a full 63-bit ``randint`` range.
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

# Where every draw is made — see the module docstring.
_HOST = "cpu"


def _as_generator(generator: object, owner: str) -> Generator | None:
    r"""Normalise a ``generator=`` argument to a :class:`lucid.Generator`.

    ``None`` stays ``None`` (draw from the default generator, which
    :func:`lucid.manual_seed` controls).  A :class:`lucid.Generator` is
    used as is and advances from epoch to epoch.  An ``int`` (or any other
    seed :class:`random.Random` accepts) is turned into one private
    generator *once*, so it reproduces the whole sequence of epochs
    rather than repeating the first epoch forever.  A ``bool`` is refused:
    ``generator=True`` reads as a switch, not as the seed 1.
    """
    if generator is None or isinstance(generator, Generator):
        return generator
    if not isinstance(generator, bool):
        if isinstance(generator, int):
            return Generator(generator % _UNSIGNED_64)
        try:
            seed = random.Random(generator).getrandbits(63)  # type: ignore[arg-type]
        except TypeError:
            pass
        else:
            return Generator(seed)
    raise TypeError(
        f"{owner}: generator must be a lucid.Generator, an int seed, or "
        f"None — got {type(generator).__name__}."
    )


def _draw_seed(generator: Generator | None) -> int:
    r"""Draw a fresh 63-bit seed from ``generator`` (default: Lucid's).

    ``lucid.randint`` rather than ``rand``: the integer draw carries the
    full 63 bits, where a float draw would collapse the seed space.
    """
    draw = randint(0, _SEED_HIGH, (1,), generator=generator, device=_HOST)
    return int(draw.item())


def _uniform_indices(n: int, count: int, generator: Generator | None) -> list[int]:
    r"""``count`` independent uniform draws from ``range(n)``."""
    if count == 0:
        return []
    draws = randint(0, n, (count,), generator=generator, device=_HOST)
    return cast(list[int], draws.tolist())


def _uniform_doubles(count: int, generator: Generator | None) -> Tensor:
    r"""``count`` uniform ``float64`` draws in ``[0, 1)``, 53 bits each, on the host.

    Built from integer draws so that every double in the grid
    ``k / 2**53`` is reachable — an inverse-CDF lookup over a long weight
    vector needs that resolution to reach its small-probability tail.
    """
    ints = randint(0, 1 << 53, (count,), generator=generator, device=_HOST)
    return ints.to(lucid.float64) * (1.0 / (1 << 53))


def _host_float64(values: list[float]) -> Tensor:
    r"""``values`` as a host ``float64`` tensor, to combine with the draws above."""
    return lucid.tensor(values, dtype=lucid.float64, device=_HOST)


def _permutation(n: int, generator: Generator | None) -> list[int]:
    r"""A uniform random permutation of ``range(n)`` as a Python list."""
    if n == 0:
        return []
    perm = randperm(n, generator=generator, device=_HOST)
    return cast(list[int], perm.tolist())
