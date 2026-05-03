"""
Worker utilities for DataLoader multi-process data loading.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from lucid.utils.data.dataset import Dataset
from typing import TYPE_CHECKING

# Thread-local storage: each worker process stores its WorkerInfo here.
_worker_local = threading.local()


@dataclass
class WorkerInfo:
    """Information about the current worker process.

    Attributes
    ----------
    id : int
        The worker's integer index in [0, num_workers).
    num_workers : int
        Total number of worker processes for this DataLoader.
    seed : int
        The per-worker random seed (base_seed + id).
    dataset : Dataset
        The dataset copy owned by this worker.
    """

    id: int
    num_workers: int
    seed: int
    dataset: Dataset


def get_worker_info() -> WorkerInfo | None:
    """Return a :class:`WorkerInfo` object in a worker process, else ``None``.

    This mirrors ``torch.utils.data.get_worker_info()``.  Inside the
    ``worker_init_fn`` or dataset ``__getitem__`` of a multi-process
    DataLoader, this function returns the WorkerInfo for the current worker.
    In the main process (or when ``num_workers=0``) it returns ``None``.
    """
    return getattr(_worker_local, "info", None)


def _set_worker_info(info: WorkerInfo | None) -> None:
    """Internal: set the WorkerInfo for the current thread/process."""
    _worker_local.info = info
