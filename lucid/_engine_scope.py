"""Scoped thread-local engine state helpers (tracer suspension and autocast dtype)."""

from collections.abc import Iterator
from contextlib import contextmanager

from lucid._C import engine as _C_engine


@contextmanager
def untraced() -> Iterator[None]:
    """Temporarily detach the active ``lucid.compile`` tracer on this thread.

    Operations executed inside this ``with`` block run eagerly without being
    recorded into the active ``Tracer`` DAG, and the previous tracer is
    restored on exit even if an exception is raised.
    """
    saved = _C_engine.compile.current_tracer()
    if saved is None:
        yield
        return
    _C_engine.compile.set_current_tracer(None)
    try:
        yield
    finally:
        _C_engine.compile.set_current_tracer(saved)


@contextmanager
def autocast_dtype(dtype: _C_engine.Dtype = _C_engine.Dtype.F32) -> Iterator[None]:
    """Pin the engine's autocast dtype to ``dtype`` for the block when autocast is active.

    Parameters
    ----------
    dtype : lucid._C.engine.Dtype, optional
        Target engine dtype to install for ``LOWER`` autocast rows
        while autocast is enabled (defaults to ``_C_engine.Dtype.F32``).
    """
    if not _C_engine.amp_is_active():
        yield
        return
    guard = _C_engine.AutocastGuard(dtype)
    try:
        yield
    finally:
        del guard
