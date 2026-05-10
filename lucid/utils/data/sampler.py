from lucid.utils.data.dataset import Dataset

"""
Sampler classes for DataLoader.
"""

from typing import Iterator
import random


class Sampler:
    """Base class for all samplers."""

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order."""

    def __init__(self, data_source: Dataset) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples elements randomly without replacement."""

    def __init__(
        self,
        data_source: Dataset,
        replacement: bool = False,
        num_samples: int | None = None,
        generator: object = None,
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

            rng = _r.Random(self.generator)  # type: ignore[arg-type]
            yield from (rng.randrange(n) for _ in range(self.num_samples))
        else:
            perm = list(range(n))
            random.shuffle(perm)
            yield from perm[: self.num_samples]

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices."""

    def __init__(self, indices: list[int], generator: object = None) -> None:
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
        generator: object = None,
    ) -> None:
        self.weights = list(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        import random as _r

        rng = _r.Random(self.generator)  # type: ignore[arg-type]
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

    def __iter__(self) -> Iterator[list[int]]:  # type: ignore[override]
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


class DistributedSampler(Sampler):
    """Subset-and-shuffle sampler for distributed training.

    Lucid is single-process, single-machine — there is no real distributed
    backend to coordinate with — but ``DistributedSampler`` is part of the
    standard ``DataLoader`` surface and user code routinely instantiates it
    even in single-rank contexts (e.g. ``num_replicas=1, rank=0``).  This
    implementation supports exactly that: it partitions the dataset into
    ``num_replicas`` slabs and yields the indices belonging to ``rank``.
    With the default ``num_replicas=1`` it degenerates to a plain
    sequential / random sampler that respects ``shuffle`` and ``seed``.

    A multi-process backend would require a process group + collective
    communication; the surface stays compatible should that land later.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank {rank} is out of range for num_replicas={num_replicas}"
            )
        self.dataset: Dataset = dataset
        self.num_replicas: int = num_replicas
        self.rank: int = rank
        self.shuffle: bool = shuffle
        self.seed: int = seed
        self.drop_last: bool = drop_last
        self.epoch: int = 0

        # Split the index range into ``num_replicas`` evenly-sized slabs.
        # With ``drop_last=False`` we wrap-pad so every replica sees the
        # same number of indices; with ``drop_last=True`` we round down.
        n: int = len(dataset)
        if drop_last:
            self.num_samples: int = n // num_replicas
            self.total_size: int = self.num_samples * num_replicas
        else:
            self.num_samples = (n + num_replicas - 1) // num_replicas
            self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number — affects the shuffling RNG seed."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        n: int = len(self.dataset)
        indices: list[int] = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            # Wrap-pad so the slab division is even.
            padding: int = self.total_size - n
            if padding > 0:
                indices += indices[:padding]
        # Slab pick: take every ``num_replicas``-th index starting at ``rank``.
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples
