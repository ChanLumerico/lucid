"""Scoped thread-local engine state helpers (tracer suspension)."""

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
