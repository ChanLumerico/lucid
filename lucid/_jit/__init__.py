"""
lucid._jit — JIT compilation surface (currently a pass-through stub).

The legacy Python tracer-based JIT (`_jit/api.py`, `_jit/executor.py`,
etc.) is obsolete now that ops execute eagerly in the C++ engine with
its own fusion / scheduling. We keep `compile()` and the
`JITFunction` / `JITModule` names so that user code that did
`@lucid.compile` continues to import — they just return the target
unchanged.

If a future C++-side JIT lands, swap the body without changing the
public surface.
"""

from __future__ import annotations

from typing import Any, Callable, overload


__all__ = ["compile", "JITFunction", "JITModule"]


class JITFunction:
    """Pass-through wrapper that mimics the legacy `JITFunction` API."""

    def __init__(self, fn: Callable, *, max_cache_entries: int = 8) -> None:
        self._fn = fn
        self._max_cache_entries = max_cache_entries

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    def invalidate_cache(self) -> None:
        pass

    def __repr__(self) -> str:
        name = getattr(self._fn, "__name__", repr(self._fn))
        return f"JITFunction({name})"


class JITModule:
    """Pass-through wrapper that mimics the legacy `JITModule` API."""

    def __init__(self, module: Any, *, max_cache_entries: int = 8) -> None:
        self._module = module
        self._max_cache_entries = max_cache_entries

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._module(*args, **kwargs)

    def invalidate_cache(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def __repr__(self) -> str:
        return f"JITModule({type(self._module).__name__})"


@overload
def compile(target: Callable, *, max_cache_entries: int = 8) -> JITFunction: ...
@overload
def compile(target: Any, *, max_cache_entries: int = 8) -> JITModule: ...
def compile(target: Any, *, max_cache_entries: int = 8) -> JITFunction | JITModule:
    """Wrap `target` in a pass-through JIT shim.

    Accepts either a plain callable or an `nn.Module`. The current
    implementation runs the target eagerly — the C++ engine handles its
    own scheduling/fusion, so a Python-level tracer would only add
    overhead.
    """
    from lucid.nn.module import Module  # local import to avoid cycle

    if isinstance(target, Module):
        return JITModule(target, max_cache_entries=max_cache_entries)
    if callable(target):
        return JITFunction(target, max_cache_entries=max_cache_entries)
    raise TypeError(
        f"lucid.compile() expects an nn.Module or a callable, got {type(target)}"
    )
