"""
Worker utilities for DataLoader multi-process data loading.
"""

import threading
from dataclasses import dataclass
from lucid.utils.data.dataset import Dataset

# Thread-local storage: each worker process stores its WorkerInfo here.
_worker_local = threading.local()


@dataclass
class WorkerInfo:
    r"""Per-worker context published inside a :class:`DataLoader` worker process.

    Returned by :func:`get_worker_info` when called from within a
    worker's ``__getitem__`` / ``__iter__`` / ``worker_init_fn``.  Lets
    user code shard work, seed RNGs differently per worker, or open
    per-worker file handles.

    Parameters
    ----------
    id : int
        The worker's integer index in ``[0, num_workers)``.
    num_workers : int
        Total number of worker processes for this :class:`DataLoader`.
    seed : int
        The per-worker random seed (typically ``base_seed + id``).  The
        loader seeds Python ``random`` and (when available) ``numpy``
        with this value before invoking ``worker_init_fn``.
    dataset : Dataset
        The dataset copy owned by this worker.  Because ``spawn`` is
        used, this is a deep-copied instance — mutations in one worker
        are not visible to others.

    Notes
    -----
    Use :func:`get_worker_info` rather than constructing this dataclass
    directly; the per-thread storage is what actually wires it up.

    Examples
    --------
    >>> def worker_init_fn(worker_id):
    ...     info = get_worker_info()
    ...     # Shard an IterableDataset across workers:
    ...     info.dataset.start = info.id * shard_size
    """

    id: int
    num_workers: int
    seed: int
    dataset: Dataset


def get_worker_info() -> WorkerInfo | None:
    r"""Return a :class:`WorkerInfo` object in a worker process, else ``None``.

    Inside the ``worker_init_fn`` or dataset ``__getitem__`` /
    ``__iter__`` of a multi-process :class:`DataLoader`, this returns
    the :class:`WorkerInfo` for the current worker.  In the main
    process — or when ``num_workers=0`` — it returns ``None``.

    Returns
    -------
    WorkerInfo or None
        Per-worker context if called from a worker, else ``None``.

    Notes
    -----
    The dominant use case is sharding an :class:`IterableDataset`
    across workers: each worker pulls its :attr:`WorkerInfo.id` and
    advances through a distinct slice of the underlying stream so that
    samples are not duplicated across the pool.

    Examples
    --------
    >>> class ShardedStream(IterableDataset):
    ...     def __iter__(self):
    ...         info = get_worker_info()
    ...         if info is None:
    ...             yield from self._all()
    ...         else:
    ...             yield from self._slice(info.id, info.num_workers)
    """
    return getattr(_worker_local, "info", None)


def _set_worker_info(info: WorkerInfo | None) -> None:
    """Internal: set the WorkerInfo for the current thread/process."""
    _worker_local.info = info
