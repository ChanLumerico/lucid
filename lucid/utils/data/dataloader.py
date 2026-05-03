"""
DataLoader and default_collate.
"""

from __future__ import annotations

import multiprocessing as _mp
import random
import threading
from typing import Any, Callable, Iterator

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

# Sentinel pushed to index queues to signal workers to shut down.
_SHUTDOWN = None


# ── collation ─────────────────────────────────────────────────────────────────


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
        return type(elem)(*(default_collate([d[i] for d in batch]) for i in range(len(elem))))

    if isinstance(elem, (list, tuple)):
        collated = [default_collate([d[i] for d in batch]) for i in range(len(elem))]
        return type(elem)(collated) if isinstance(elem, tuple) else collated

    return batch


# ── worker process entry point ────────────────────────────────────────────────
# Must be a top-level function so `spawn` can pickle it.

def _worker_loop(
    worker_id: int,
    dataset: Dataset,
    index_queue: Any,
    result_queue: Any,
    collate_fn: Callable,
    worker_init_fn: Callable | None,
    seed: int,
) -> None:
    """Worker process: pull index batches, fetch data, push collated results."""
    # Seed this worker independently for reproducibility.
    random.seed(seed)
    np.random.seed(seed % (2**32))

    if worker_init_fn is not None:
        worker_init_fn(worker_id)

    while True:
        msg = index_queue.get()
        if msg is _SHUTDOWN:
            return
        seq_num, indices = msg
        try:
            batch = [dataset[i] for i in indices]
            result = collate_fn(batch)
            result_queue.put((seq_num, result))
        except Exception as exc:  # noqa: BLE001
            result_queue.put((seq_num, exc))


# ── single-process iterator ───────────────────────────────────────────────────


class _SingleProcessDataLoaderIter:
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


# ── multi-process iterator ────────────────────────────────────────────────────


class _MultiProcessDataLoaderIter:
    """Multi-worker iterator with prefetching and ordered delivery.

    Design:
    - Each worker owns one index queue; main process round-robins index batches.
    - Workers push (seq_num, batch) onto a single shared result queue.
    - Main process reorders via a dict buffer, yielding batches in original order.
    - Prefetch depth: num_workers × prefetch_factor batches kept in flight.
    """

    def __init__(self, loader: DataLoader) -> None:
        self._num_workers: int = loader.num_workers
        self._prefetch_factor: int = loader.prefetch_factor or 2
        self._persistent: bool = loader.persistent_workers
        self._timeout: float = loader.timeout

        # multiprocessing_context: use caller's choice if provided,
        # otherwise default to 'spawn' (safe on macOS/Apple Silicon).
        mp_ctx = loader.multiprocessing_context
        if mp_ctx is None:
            ctx = _mp.get_context("spawn")
        elif isinstance(mp_ctx, str):
            ctx = _mp.get_context(mp_ctx)
        else:
            ctx = mp_ctx  # already a context object

        # One index queue per worker to avoid contention.
        self._index_queues = [ctx.Queue() for _ in range(self._num_workers)]
        self._result_queue: Any = ctx.Queue()

        # Sequence counters.
        self._send_idx: int = 0    # next batch index to dispatch
        self._rcvd_idx: int = 0    # next batch index to yield
        self._reorder: dict[int, Any] = {}

        # Materialise the full batch list once per epoch.
        self._batches: list[list[int]] = list(loader.batch_sampler)
        self._n_batches: int = len(self._batches)

        base_seed = random.randint(0, 2**31)
        self._workers = [
            ctx.Process(
                target=_worker_loop,
                args=(
                    wid,
                    loader.dataset,
                    self._index_queues[wid],
                    self._result_queue,
                    loader.collate_fn,
                    loader.worker_init_fn,
                    base_seed + wid,
                ),
                daemon=True,
            )
            for wid in range(self._num_workers)
        ]
        for w in self._workers:
            w.start()

        # Prefill the pipeline.
        prefill = min(self._num_workers * self._prefetch_factor, self._n_batches)
        for _ in range(prefill):
            self._dispatch_next()

    # ── internal helpers ───────────────────────────────────────────────────────

    def _dispatch_next(self) -> None:
        if self._send_idx >= self._n_batches:
            return
        worker_id = self._send_idx % self._num_workers
        self._index_queues[worker_id].put((self._send_idx, self._batches[self._send_idx]))
        self._send_idx += 1

    def _shutdown_workers(self) -> None:
        for q in self._index_queues:
            q.put(_SHUTDOWN)
        for w in self._workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()

    # ── iteration ─────────────────────────────────────────────────────────────

    def __iter__(self) -> _MultiProcessDataLoaderIter:
        return self

    def __next__(self) -> Any:
        if self._rcvd_idx >= self._n_batches:
            if not self._persistent:
                self._shutdown_workers()
            raise StopIteration

        # Collect from the result queue until the in-order batch is ready.
        # Apply timeout if set (>0), otherwise block indefinitely.
        get_kwargs = {"timeout": self._timeout} if self._timeout > 0 else {}
        while self._rcvd_idx not in self._reorder:
            try:
                seq, result = self._result_queue.get(**get_kwargs)
            except Exception:  # queue.Empty on timeout
                self._shutdown_workers()
                raise RuntimeError(
                    f"DataLoader worker timed out after {self._timeout}s. "
                    "Increase timeout or reduce batch size."
                ) from None
            if isinstance(result, Exception):
                self._shutdown_workers()
                raise result
            self._reorder[seq] = result

        batch = self._reorder.pop(self._rcvd_idx)
        self._rcvd_idx += 1
        self._dispatch_next()   # keep the pipeline full
        return batch

    def __del__(self) -> None:
        try:
            self._shutdown_workers()
        except Exception:  # noqa: BLE001
            pass


# ── DataLoader ────────────────────────────────────────────────────────────────


class DataLoader:
    """Combine a dataset with a sampler to provide iteration over mini-batches.

    Parameters
    ----------
    dataset : Dataset
        Dataset to load data from.
    batch_size : int, optional
        Samples per batch (default: 1).
    shuffle : bool, optional
        Shuffle at the start of each epoch (default: False).
    sampler : Sampler, optional
        Custom index sampler.
    batch_sampler : Sampler, optional
        Custom batch sampler.
    num_workers : int, optional
        Worker processes for parallel data loading. ``0`` = single-process.
    collate_fn : callable, optional
        Merge a list of samples into a batch.
    pin_memory : bool, optional
        No-op on Apple Silicon (unified memory).
    drop_last : bool, optional
        Drop the last incomplete batch (default: False).
    timeout : float, optional
        Timeout for collecting a batch from workers (default: 0, unlimited).
    worker_init_fn : callable, optional
        Called as ``worker_init_fn(worker_id)`` at the start of each worker.
    prefetch_factor : int, optional
        Batches pre-loaded per worker (default: 2). Only used when
        ``num_workers > 0``.
    persistent_workers : bool, optional
        Keep worker processes alive between epochs (default: False).
    generator : optional
        RNG for random sampler.

    Examples
    --------
    >>> ds = TensorDataset(X, y)
    >>> loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
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
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ) -> None:
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        if prefetch_factor is not None and prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be > 0, got {prefetch_factor}")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn or default_collate
        self.pin_memory = pin_memory          # no-op on Apple Silicon
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.generator = generator
        # Match PyTorch: prefetch_factor=None when num_workers=0, else default 2
        if prefetch_factor is None:
            self.prefetch_factor = 2 if num_workers > 0 else None
        else:
            self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler is mutually exclusive with "
                    "batch_size, shuffle, sampler, and drop_last."
                )
            self.batch_sampler = batch_sampler
            self.batch_size = None  # type: ignore[assignment]
        else:
            if sampler is not None and shuffle:
                raise ValueError("sampler and shuffle are mutually exclusive.")
            if sampler is None:
                sampler = (
                    RandomSampler(dataset, generator=generator)
                    if shuffle  # None and False both → SequentialSampler
                    else SequentialSampler(dataset)
                )
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self._persistent_iter: _MultiProcessDataLoaderIter | None = None

    def __iter__(self) -> Iterator[Any]:
        if self.num_workers == 0:
            yield from _SingleProcessDataLoaderIter(self)
            return

        if self.persistent_workers:
            if self._persistent_iter is None:
                self._persistent_iter = _MultiProcessDataLoaderIter(self)
            else:
                # Reset counters for a new epoch while workers stay alive.
                it = self._persistent_iter
                it._batches = list(self.batch_sampler)
                it._n_batches = len(it._batches)
                it._send_idx = 0
                it._rcvd_idx = 0
                it._reorder.clear()
                prefill = min(self.num_workers * (self.prefetch_factor or 2), it._n_batches)
                for _ in range(prefill):
                    it._dispatch_next()
            yield from self._persistent_iter
        else:
            yield from _MultiProcessDataLoaderIter(self)

    def __len__(self) -> int:
        return len(self.batch_sampler)
