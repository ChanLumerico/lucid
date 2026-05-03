"""
Sampler classes for DataLoader.
"""

from typing import Any, Iterator
import random


class Sampler:
    """Base class for all samplers."""

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order."""

    def __init__(self, data_source: Any) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples elements randomly without replacement."""

    def __init__(
        self,
        data_source: Any,
        replacement: bool = False,
        num_samples: int | None = None,
        generator: Any = None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

    @property
    def num_samples(self) -> int:
        return (
            self._num_samples
            if self._num_samples is not None
            else len(self.data_source)
        )

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            import random as _r

            rng = _r.Random(self.generator)
            yield from (rng.randrange(n) for _ in range(self.num_samples))
        else:
            perm = list(range(n))
            random.shuffle(perm)
            yield from perm[: self.num_samples]

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices."""

    def __init__(self, indices: list[int], generator: Any = None) -> None:
        self.indices = list(indices)
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        perm = list(self.indices)
        random.shuffle(perm)
        yield from perm

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    """Samples elements with given probabilities (weights)."""

    def __init__(
        self,
        weights: list[float],
        num_samples: int,
        replacement: bool = True,
        generator: Any = None,
    ) -> None:
        self.weights = list(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        import random as _r

        rng = _r.Random(self.generator)
        total = sum(self.weights)
        normalized = [w / total for w in self.weights]
        n = len(self.weights)
        if self.replacement:
            for _ in range(self.num_samples):
                r = rng.random()
                cumulative = 0.0
                for i, w in enumerate(normalized):
                    cumulative += w
                    if r <= cumulative:
                        yield i
                        break
        else:
            # Reservoir sampling (simple approach without replacement)
            indices = list(range(n))
            chosen = _r.choices(indices, weights=self.weights, k=self.num_samples)
            yield from chosen

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler):
    """Wrap a sampler to yield mini-batches of indices."""

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
