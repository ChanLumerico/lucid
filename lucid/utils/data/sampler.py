from lucid.utils.data.dataset import Dataset

"""
Sampler classes for DataLoader.
"""

from typing import Iterator
import random


class Sampler:
    """Abstract base class for index samplers used by :class:`DataLoader`.

    A sampler iterates over integer indices into a map-style dataset.
    Subclasses must implement :meth:`__iter__` (yielding indices) and
    :meth:`__len__` (total number of indices produced per epoch).

    Notes
    -----
    Samplers decouple *which* samples are visited from *how* they are
    fetched — the dataset answers ``__getitem__(i)`` and the sampler
    decides the sequence of ``i`` values.  This separation is what lets
    :class:`DataLoader` swap iteration policies (sequential / random /
    weighted / distributed) without touching the dataset.

    Examples
    --------
    >>> class Even(Sampler):
    ...     def __init__(self, n): self.n = n
    ...     def __iter__(self): return iter(range(0, self.n, 2))
    ...     def __len__(self): return (self.n + 1) // 2
    >>> list(Even(6))
    [0, 2, 4]
    """

    def __iter__(self) -> Iterator[int]:
        """Yield integer sample indices.

        Returns
        -------
        Iterator[int]
            Iterator over ``0 <= i < len(data_source)`` values, in an
            order defined by the concrete sampler.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of indices yielded in one full pass."""
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Yield indices in fixed order ``0, 1, ..., len(data_source) - 1``.

    The default sampler used when ``shuffle=False`` and no explicit
    sampler is supplied to :class:`DataLoader`.

    Parameters
    ----------
    data_source : Dataset
        Dataset whose ``__len__`` determines the index range.

    Notes
    -----
    Order is fully deterministic — no RNG is consulted — so two passes
    over the sampler always yield the same index sequence.  This is the
    right choice for evaluation / inference loops where ordering must
    line up with external bookkeeping (e.g. per-row metric arrays).

    Examples
    --------
    >>> sampler = SequentialSampler(my_dataset)
    >>> list(sampler)[:5]
    [0, 1, 2, 3, 4]
    """

    def __init__(self, data_source: Dataset) -> None:
        """Store ``data_source`` for length introspection.

        Parameters
        ----------
        data_source : Dataset
            Dataset to be iterated sequentially.
        """
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        """Yield ``0, 1, ..., len(data_source) - 1`` in order."""
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        """Return ``len(data_source)``."""
        return len(self.data_source)


class RandomSampler(Sampler):
    """Yield indices in random order, with or without replacement.

    Parameters
    ----------
    data_source : Dataset
        Dataset whose ``__len__`` defines the index range ``[0, n)``.
    replacement : bool, optional
        If ``True``, indices are drawn with replacement (any index can
        appear multiple times). If ``False`` (default), indices are a
        random permutation of ``range(n)`` and each index appears exactly
        once per epoch.
    num_samples : int, optional
        Number of indices to draw per epoch. Defaults to
        ``len(data_source)``. Only meaningful when ``replacement=True``;
        otherwise it truncates the permutation.
    generator : optional
        Seed-like object forwarded to ``random.Random`` for reproducibility
        when ``replacement=True``.

    Notes
    -----
    Without replacement, each epoch is a fresh uniform permutation of
    ``range(n)`` — equivalent to shuffling.  With replacement, indices
    are i.i.d. uniform draws and ``num_samples`` controls the per-epoch
    budget independently of ``n``; this lets the caller oversample
    (``num_samples > n``) for stochastic training regimes.

    Examples
    --------
    >>> sampler = RandomSampler(my_dataset)
    >>> for idx in sampler:
    ...     x = my_dataset[idx]
    """

    def __init__(
        self,
        data_source: Dataset,
        replacement: bool = False,
        num_samples: int | None = None,
        generator: object = None,
    ) -> None:
        """Configure the random sampler.

        Parameters
        ----------
        data_source : Dataset
            Source dataset.
        replacement : bool
            See class docstring.
        num_samples : int, optional
            See class docstring.
        generator : optional
            See class docstring.
        """
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

    @property
    def num_samples(self) -> int:
        """Effective number of samples per epoch.

        Returns the explicit ``num_samples`` argument if provided,
        otherwise ``len(data_source)``.
        """
        return (
            self._num_samples
            if self._num_samples is not None
            else len(self.data_source)
        )

    def __iter__(self) -> Iterator[int]:
        """Yield ``num_samples`` indices according to the configured strategy.

        With ``replacement=True`` each index is drawn uniformly with
        replacement using a seeded ``random.Random``. With
        ``replacement=False`` a fresh permutation of ``range(n)`` is
        produced via the module-level ``random.shuffle`` and truncated
        to ``num_samples``.
        """
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
        """Return :attr:`num_samples`."""
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Yield a random permutation of a fixed list of indices each epoch.

    Useful when the caller already knows which subset of the dataset
    should be visited (e.g., a precomputed train/val split) and only
    wants shuffling among that subset.

    Parameters
    ----------
    indices : list of int
        Indices into the parent dataset to sample from.
    generator : optional
        Seed-like object accepted for API compatibility; the current
        implementation defers to the global ``random`` state.

    Notes
    -----
    The index pool itself is fixed at construction time; only the
    *order* changes between epochs.  Useful with precomputed
    cross-validation folds — store the per-fold index list once, then
    instantiate one :class:`SubsetRandomSampler` per fold.

    Examples
    --------
    >>> fold_indices = [3, 5, 7, 9, 11]
    >>> sampler = SubsetRandomSampler(fold_indices)
    >>> sorted(list(sampler)) == fold_indices
    True
    """

    def __init__(self, indices: list[int], generator: object = None) -> None:
        """Store the index pool and optional generator handle.

        Parameters
        ----------
        indices : list of int
            Indices to sample from.
        generator : optional
            Accepted for API compatibility.
        """
        self.indices = list(indices)
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        """Yield ``self.indices`` in a freshly shuffled order each epoch."""
        perm = list(self.indices)
        random.shuffle(perm)
        yield from perm

    def __len__(self) -> int:
        """Return the size of the index pool."""
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    r"""Yield indices drawn proportionally to user-supplied weights.

    Each index ``i`` is selected with probability proportional to
    ``weights[i]``. Internally the weights are normalised by their sum so
    they need not form a true probability distribution on input.

    Parameters
    ----------
    weights : list of float
        Non-negative weight per index. The effective probability of index
        ``i`` is :math:`p_i = w_i / \sum_j w_j`.
    num_samples : int
        Number of indices to draw per epoch.
    replacement : bool, optional
        If ``True`` (default), draws are independent with replacement —
        the same index may appear multiple times. If ``False``, draws use
        weighted reservoir-style selection (via ``random.choices``); note
        that the underlying call here still permits repeats, so callers
        wanting strict-uniqueness should provide ``num_samples <=
        len(weights)`` and validate downstream.
    generator : optional
        Seed-like object forwarded to ``random.Random`` for reproducibility.

    Notes
    -----
    Each index :math:`i` is selected with probability

    .. math::

        P(i) = \frac{w_i}{\sum_j w_j},

    so the user-supplied weights need not be normalised.  The classic
    use case is *class-imbalance correction*: set ``w_i = 1 /
    class_count[label_i]`` so under-represented classes are upsampled
    to roughly uniform frequency.  Sampling with replacement is the
    default — it preserves the target marginal exactly and is the only
    fully consistent option when ``num_samples`` exceeds the number of
    nonzero-weight indices.

    Examples
    --------
    >>> # 3 classes, counts [900, 90, 10]; upweight rare classes
    >>> weights = [1/900]*900 + [1/90]*90 + [1/10]*10
    >>> sampler = WeightedRandomSampler(weights, num_samples=1000)
    >>> for idx in sampler:
    ...     x, y = my_dataset[idx]
    """

    def __init__(
        self,
        weights: list[float],
        num_samples: int,
        replacement: bool = True,
        generator: object = None,
    ) -> None:
        """Store sampling configuration.

        Parameters
        ----------
        weights : list of float
            See class docstring.
        num_samples : int
            See class docstring.
        replacement : bool
            See class docstring.
        generator : optional
            See class docstring.
        """
        self.weights = list(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        """Yield ``num_samples`` indices proportional to the weights.

        With ``replacement=True`` indices are produced by inverse-CDF
        sampling against the normalised weight vector. With
        ``replacement=False`` ``random.choices`` is used with the raw
        weights.
        """
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
        """Return :attr:`num_samples`."""
        return self.num_samples


class BatchSampler(Sampler):
    """Group an inner sampler's indices into mini-batches.

    Wraps an existing :class:`Sampler` and packs its emitted indices into
    fixed-size lists. This is what :class:`DataLoader` uses internally to
    convert a per-sample index stream into per-batch index lists.

    Parameters
    ----------
    sampler : Sampler
        Underlying per-sample index sampler.
    batch_size : int
        Number of indices per yielded batch.
    drop_last : bool
        If ``True``, drop the trailing batch when the inner sampler's
        length is not divisible by ``batch_size``. If ``False``, yield
        the short final batch.

    Notes
    -----
    The inner sampler is consumed lazily — :class:`BatchSampler` simply
    accumulates ``batch_size`` indices then yields the list and starts a
    new batch.  ``drop_last=True`` produces ``floor(n / batch_size)``
    batches of uniform size (the usual choice for training, where
    short batches mess with BatchNorm statistics); ``drop_last=False``
    produces ``ceil(n / batch_size)`` batches with the final one
    possibly short (the usual choice for evaluation, where every sample
    must be visited).

    Examples
    --------
    >>> inner = SequentialSampler(my_dataset)   # 10 items
    >>> bs = BatchSampler(inner, batch_size=4, drop_last=False)
    >>> [b for b in bs]
    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        """Store the inner sampler and batching configuration.

        Parameters
        ----------
        sampler : Sampler
            Inner per-sample sampler.
        batch_size : int
            Target batch size.
        drop_last : bool
            Whether to discard the final short batch.
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:  # type: ignore[override]
        """Yield successive batches of indices.

        Yields
        ------
        list of int
            Lists of length ``batch_size``, except possibly the last
            (which is dropped if ``drop_last`` is ``True``).
        """
        batch: list[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches produced per epoch.

        Computed as ``floor(n / batch_size)`` when ``drop_last`` is
        ``True``, else ``ceil(n / batch_size)``.
        """
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

    Notes
    -----
    The index range ``range(len(dataset))`` is partitioned into
    ``num_replicas`` interleaved slabs — rank ``r`` receives every
    ``num_replicas``-th index starting at ``r``.  Per-epoch shuffles
    are driven by ``random.Random(seed + epoch)``, so every replica
    sees a different slab while remaining globally deterministic.
    Call :meth:`set_epoch` once per epoch to rotate the shuffle —
    forgetting to do so produces identical orderings each pass.

    Examples
    --------
    >>> sampler = DistributedSampler(my_dataset, num_replicas=4, rank=2)
    >>> for epoch in range(num_epochs):
    ...     sampler.set_epoch(epoch)
    ...     for idx in sampler:
    ...         x = my_dataset[idx]
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
        """Configure the distributed sampler.

        Parameters
        ----------
        dataset : Dataset
            Dataset whose ``__len__`` defines the index range.
        num_replicas : int, optional
            Number of participating replicas (default ``1`` — degenerate
            single-process case).
        rank : int, optional
            Replica id in ``[0, num_replicas)``.
        shuffle : bool, optional
            If ``True``, indices are shuffled by ``random.Random(seed +
            epoch)`` before slabbing.
        seed : int, optional
            Base seed for the shuffling RNG. Combined with
            :meth:`set_epoch` for deterministic per-epoch shuffles.
        drop_last : bool, optional
            If ``True``, drop the trailing remainder so every replica
            sees the same number of samples without padding. If
            ``False``, wrap-pad the index list.

        Raises
        ------
        ValueError
            If ``num_replicas < 1`` or ``rank`` is out of range.
        """
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
        """Yield this replica's slab of indices for the current epoch.

        Indices are optionally shuffled with seed ``self.seed + self.epoch``,
        then either truncated to ``total_size`` (``drop_last=True``) or
        wrap-padded (``drop_last=False``). The replica picks every
        ``num_replicas``-th index starting at ``rank``.
        """
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
        """Return per-replica sample count — same on every rank."""
        return self.num_samples
