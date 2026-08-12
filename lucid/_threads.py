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

    ``n = 0`` restores the "library default" sentinel, which is what
    :func:`get_num_threads` reports before anything has been set.  The
    setter has to accept it: a getter whose value its own setter rejects
    cannot be snapshotted and put back, and every save/restore around
    this pair — checkpointing, test fixtures, the audit's state guard —
    silently leaves the count changed instead.
    """
    global _intra_op_threads
    if int(n) < 0:
        raise ValueError(f"set_num_threads requires n >= 0, got {n}")
    _intra_op_threads = int(n)


def get_num_threads() -> int:
    """Return the last value passed to :func:`set_num_threads` (or 0 if
    never set, meaning the underlying library's default)."""
    return _intra_op_threads


def set_num_interop_threads(n: int) -> None:
    """Record the desired inter-op thread count.  Advisory only — Lucid
    does not maintain a separate inter-op pool.

    ``n = 0`` restores the "library default" sentinel, for the same
    round-trip reason as :func:`set_num_threads`.
    """
    global _inter_op_threads
    if int(n) < 0:
        raise ValueError(f"set_num_interop_threads requires n >= 0, got {n}")
    _inter_op_threads = int(n)


def get_num_interop_threads() -> int:
    """Return the recorded inter-op thread count, or ``0`` if unset.

    Returns the value most recently passed to
    :func:`set_num_interop_threads` — purely informational, since
    Lucid does not maintain a separate inter-op thread pool (the
    setter is API parity, not a switch).

    Returns
    -------
    int
        Last recorded count, or ``0`` when none has been set.
    """
    return _inter_op_threads


__all__ = [
    "set_num_threads",
    "get_num_threads",
    "set_num_interop_threads",
    "get_num_interop_threads",
]
