"""
lucid.compile._fallback — Phase 1.4 eager-escape helpers.

When :class:`CompiledModule` cannot compile a given signature (a
forward that touches an op with no emitter, or that breaks an
invariant of the MPSGraph builder), it must fall back to the regular
eager forward.  This module factors the bookkeeping out of
:class:`CompiledModule` so the wrapper stays focused on the cache /
dispatch logic.

Two responsibilities:

1. Provide a :func:`run_eager` helper that calls ``model(*args,
   **kwargs)`` exactly the way the user would (no special arguments,
   no monkey-patching).
2. Maintain a per-:class:`CompiledModule` *blacklist* of signatures
   that already failed to compile, so we don't re-trace + re-attempt
   on every call.
"""

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from lucid.compile._signature import CacheKey
    from lucid.nn.module import Module

__all__ = ["run_eager", "EagerFallbackSet"]


def run_eager(
    model: Module, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    """Invoke ``model(*args, **kwargs)`` on the eager path.

    Thin wrapper to keep the call-site self-documenting and easy to
    instrument later (timing, counters).

    Parameters
    ----------
    model : Module
        The callable to invoke.  Almost always a ``Module`` subclass,
        but the parameter is typed loosely so call sites can hand in
        any plain callable.
    args : tuple
        Positional arguments to forward.
    kwargs : dict
        Keyword arguments to forward.

    Returns
    -------
    object
        Whatever ``model`` returns — including non-tensor structured
        outputs (lists, dicts, dataclasses).
    """

    return model(*args, **kwargs)


class EagerFallbackSet:
    """Track signatures that already failed to compile.

    A signature lands in here when :func:`MpsBuilder.compile_trace…`
    returns nullptr (eager-only) or when tracing itself raises.  All
    future calls with the same key skip the compile attempt entirely
    and route straight to :func:`run_eager`.

    Cleared by :meth:`CompiledModule.clear_cache` so the user can opt
    back into a compile retry after a deliberate model change.
    """

    def __init__(self) -> None:
        self._sigs: set[CacheKey] = set()

    def add(self, key: CacheKey) -> None:
        self._sigs.add(key)

    def __contains__(self, key: CacheKey) -> bool:
        return key in self._sigs

    def __len__(self) -> int:
        return len(self._sigs)

    def clear(self) -> None:
        self._sigs.clear()

    def snapshot(self) -> tuple[CacheKey, ...]:
        # Deterministic ordering for cache_info / debugging.  ``hash``
        # is content-based on the frozen dataclass so the sort is
        # stable across calls within a session.
        return tuple(sorted(self._sigs, key=hash))


def make_eager_runner(model: Module) -> Callable[..., object]:
    """Return a no-arg-binding callable that defers to ``model(*args, **kwargs)``.

    Useful when a caller wants to capture the *current* eager forward
    once and reuse it (e.g. for benchmarking).  Reading ``model`` at
    call time means re-binding follows the user's ``_apply`` / ``to``
    mutations.
    """

    def _runner(*args: object, **kwargs: object) -> object:
        return run_eager(model, args, kwargs)

    return _runner
