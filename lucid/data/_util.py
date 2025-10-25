from typing import Sequence
import random
import math

import lucid

from ._base import Dataset, Subset


__all__ = ["random_split"]


def _resolve_lengths_from_fractions(fractions: Sequence[float], n: int) -> list[int]:
    if not fractions:
        raise ValueError("fractions must be non-empty.")
    if any(f < 0 for f in fractions):
        raise ValueError("Fractional lengths mus be non-negative.")

    s = sum(fractions)
    if not math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"When passing fractions, they must sum to 1.0 (got {s}).")

    base = [int(math.floor(f * n)) for f in fractions]
    remainder = n - sum(base)
    for i in range(remainder):
        base[i % len(base)] += 1

    return base


def random_split(
    dataset: Dataset, lengths: Sequence[int | float], seed: int | None = None
) -> tuple[Subset, ...]:
    n = len(dataset)
    if not lengths:
        raise ValueError("lengths must be non-empty.")

    all_int = all(isinstance(l, int) for l in lengths)
    all_float = all(isinstance(l, float) for l in lengths)

    if not (all_int or all_float):
        return TypeError("lengths must be all integers or all floats.")

    if all_float:
        int_lengths = _resolve_lengths_from_fractions(lengths, n)
    else:
        int_lengths = list(lengths)
        s = sum(int_lengths)
        if s != n:
            raise ValueError(
                f"Sum of input lengths ({s}) does not equal dataset length ({n})."
            )
        if any(l < 0 for l in int_lengths):
            raise ValueError("All split lengths must be non-negative.")

    if seed is None:
        seed = lucid.random.get_seed()
    rng = random.Random(seed)

    indices = list(range(n))
    rng.shuffle(indices)

    splits: list[Subset] = []
    offset = 0
    for length in int_lengths:
        split_idx = indices[offset : offset + length]
        splits.append(Subset(dataset, split_idx))
        offset += length

    return tuple(splits)
