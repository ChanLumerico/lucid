"""Thread-count getter / setter stubs for the parallelism portion of the
standard reference framework's API surface.

Lucid's CPU kernels delegate to Apple Accelerate (vDSP / vForce / BLAS /
LAPACK), whose threading is configured via the ``VECLIB_MAXIMUM_THREADS``
environment variable at process start — *not* via runtime API calls.
The MLX (GPU) path schedules its own work onto Metal command queues and
likewise has no Python-tunable thread pool.

These stubs exist so that user code which sprinkles
``lucid.set_num_threads(n)`` for portability still imports cleanly; the
recorded value is purely advisory and is not propagated to any kernel.
``get_num_threads`` returns whatever was last set (default = 0, meaning
"library default").  ``get_num_interop_threads`` mirrors this for the
inter-op pool that Lucid does not maintain.

To actually tune Accelerate threading, set ``VECLIB_MAXIMUM_THREADS`` in
the shell environment before launching Python.
"""

# Module-level stash of the advisory thread counts.  Defaults of 0 match
# the standard reference framework's "let the runtime decide" sentinel.
_intra_op_threads: int = 0
_inter_op_threads: int = 0


def set_num_threads(n: int) -> None:
    """Record the desired number of intra-op threads.  Advisory only —
    Accelerate threading is configured via ``VECLIB_MAXIMUM_THREADS`` at
    process start, not via runtime API.
    """
    global _intra_op_threads
    if int(n) <= 0:
        raise ValueError(f"set_num_threads requires n >= 1, got {n}")
    _intra_op_threads = int(n)


def get_num_threads() -> int:
    """Return the last value passed to :func:`set_num_threads` (or 0 if
    never set, meaning the underlying library's default)."""
    return _intra_op_threads


def set_num_interop_threads(n: int) -> None:
    """Record the desired inter-op thread count.  Advisory only — Lucid
    does not maintain a separate inter-op pool."""
    global _inter_op_threads
    if int(n) <= 0:
        raise ValueError(f"set_num_interop_threads requires n >= 1, got {n}")
    _inter_op_threads = int(n)


def get_num_interop_threads() -> int:
    """Return the last value passed to :func:`set_num_interop_threads`."""
    return _inter_op_threads


__all__ = [
    "set_num_threads",
    "get_num_threads",
    "set_num_interop_threads",
    "get_num_interop_threads",
]
