"""
DataLoader and default_collate.
"""

from typing import Any, Callable, Iterator, TYPE_CHECKING
import numpy as np
from lucid._tensor.tensor import Tensor
from lucid._factories.converters import tensor as _tensor_fn
from lucid._ops import stack

from lucid.utils.data.dataset import Dataset, IterableDataset
from lucid.utils.data.sampler import (
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)

if TYPE_CHECKING:
    pass


def default_collate(batch: list[Any]) -> Any:
    """Collate a list of samples into a batched tensor or nested structure."""
    elem = batch[0]

    if isinstance(elem, Tensor):
        return stack(batch, 0)

    if isinstance(elem, np.ndarray):
        return Tensor(np.stack(batch, axis=0))

    if isinstance(elem, (int, float)):
        arr = np.array(batch, dtype=np.float32 if isinstance(elem, float) else np.int64)
        return Tensor(arr)

    if isinstance(elem, (str, bytes)):
        return batch

    if isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}

    if isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return type(elem)(*[default_collate([d[i] for d in batch]) for i in range(len(elem))])

    if isinstance(elem, (list, tuple)):
        collated = [default_collate([d[i] for d in batch]) for i in range(len(elem))]
        return type(elem)(collated) if isinstance(elem, tuple) else collated

    return batch


class _SingleProcessDataLoaderIter:
    """Single-process data loading iterator."""

    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._collate_fn = loader.collate_fn
        self._batch_sampler = loader.batch_sampler
        self._iter = iter(self._batch_sampler)

    def __iter__(self) -> _SingleProcessDataLoaderIter:
        return self

    def __next__(self) -> Any:
        indices = next(self._iter)
        batch = [self._dataset[i] for i in indices]
        return self._collate_fn(batch)


class DataLoader:
    """Combine a dataset with a sampler to provide iteration over mini-batches.

    Parameters
    ----------
    dataset : Dataset
        Dataset to load data from.
    batch_size : int, optional
        Samples per batch (default: 1).
    shuffle : bool, optional
        Shuffle at the start of each epoch.
    sampler : Sampler, optional
        Custom index sampler.
    batch_sampler : Sampler, optional
        Custom batch sampler.
    num_workers : int, optional
        Number of worker processes. ``0`` means single-process.
    collate_fn : callable, optional
        Merges a list of samples into a batch.
    pin_memory : bool, optional
        No-op on Apple Silicon.
    drop_last : bool, optional
        Drop the last incomplete batch (default: ``False``).

    Examples
    --------
    >>> ds = TensorDataset(X, y)
    >>> loader = DataLoader(ds, batch_size=32, shuffle=True)
    >>> for x_batch, y_batch in loader:
    ...     loss = model(x_batch).mean()
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool | None = None,
        sampler: Sampler | None = None,
        batch_sampler: Sampler | None = None,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0.0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Any = None,
        generator: Any = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn or default_collate
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.generator = generator

        if num_workers > 0:
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")

        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler is mutually exclusive with batch_size, "
                    "shuffle, sampler, and drop_last."
                )
            self.batch_sampler = batch_sampler
            self.batch_size = None  # type: ignore[assignment]
        else:
            if sampler is not None and shuffle:
                raise ValueError("sampler and shuffle are mutually exclusive.")
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)
                else:
                    sampler = SequentialSampler(dataset)
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler

    def __iter__(self) -> Iterator[Any]:
        if self.num_workers == 0:
            yield from _SingleProcessDataLoaderIter(self)
        else:
            yield from self._multiprocess_iter()

    def _multiprocess_iter(self) -> Iterator[Any]:
        """Multi-process loading using spawn context (macOS safe)."""
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")

        indices_batches = list(self.batch_sampler)
        chunk = max(1, len(indices_batches) // self.num_workers)

        def _worker_fn(
            batch_indices: list[list[int]],
            result_queue: Any,
            dataset: Dataset,
            collate_fn: Callable,
        ) -> None:
            for indices in batch_indices:
                batch = [dataset[i] for i in indices]
                result_queue.put(collate_fn(batch))
            result_queue.put(None)

        result_queue: Any = ctx.Queue()
        workers = []
        for i in range(0, len(indices_batches), chunk):
            chunk_batches = indices_batches[i: i + chunk]
            p = ctx.Process(
                target=_worker_fn,
                args=(chunk_batches, result_queue, self.dataset, self.collate_fn),
            )
            p.start()
            workers.append(p)

        done_count = 0
        while done_count < len(workers):
            item = result_queue.get()
            if item is None:
                done_count += 1
            else:
                yield item

        for p in workers:
            p.join()

    def __len__(self) -> int:
        return len(self.batch_sampler)
