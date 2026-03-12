"""
lucid.compile._compiled
-----------------------
Compiled callable wrapper and execution engine.

Overview
~~~~~~~~
:class:`CompiledCallable` wraps any Python callable (an ``nn.Module`` or a
plain function) and provides a *compile-once, run-many* execution path:

1. **First call** – the wrapper runs the callable eagerly inside
   :func:`~lucid.compile._trace.tracing_mode` to capture the full computation
   graph as an :class:`~lucid.compile._ir.IRGraph`.  It then applies the
   requested optimisation passes and stores the result in a shape-keyed cache.

2. **Subsequent calls** – provided the input shapes match a cached entry the
   optimised graph is re-used directly.  The actual *execution* still delegates
   to the original callable so that Lucid's autograd remains fully functional;
   the graph is used only for introspection and optimisation feedback.

Shape cache
~~~~~~~~~~~
The cache key is the tuple of ``(shape, dtype)`` pairs for all input tensors.
When ``dynamic=True`` shape changes force re-compilation instead of raising an
error; when ``dynamic=False`` a mismatch raises :class:`ShapeMismatchError`.

Compile modes
~~~~~~~~~~~~~
============  ===============================================================
``"default"``         DCE + operator fusion (always safe)
``"reduce-overhead"`` Same passes; skip Python Module dispatch on cache hits
``"max-autotune"``    DCE + constant folding + operator fusion (eval only)
============  ===============================================================
"""

from __future__ import annotations

import time
from typing import Any, Callable

import lucid
from lucid.compile._ir import IRGraph
from lucid.compile._trace import tracing_mode
from lucid.compile._passes import run_default_passes, run_max_passes

__all__ = ["CompiledCallable", "CompilationResult", "ShapeMismatchError"]


class ShapeMismatchError(RuntimeError):
    """Raised when input shapes change and ``dynamic=False``."""


class CompilationResult:
    """Holds the artefacts produced by one compilation run.

    Attributes
    ----------
    graph:
        The optimised :class:`~lucid.compile._ir.IRGraph`.
    input_key:
        Cache key ``tuple[(shape, dtype), ...]`` used to look up this entry.
    compile_time_ms:
        Wall-clock time spent in the compilation step (tracing + passes).
    mode:
        The compile mode string that was active.
    """

    def __init__(
        self,
        graph: IRGraph,
        input_key: tuple,
        compile_time_ms: float,
        mode: str,
    ) -> None:
        self.graph = graph
        self.input_key = input_key
        self.compile_time_ms = compile_time_ms
        self.mode = mode

    def __repr__(self) -> str:
        return (
            f"CompilationResult("
            f"mode={self.mode!r}, "
            f"graph={self.graph}, "
            f"compile_time={self.compile_time_ms:.2f}ms)"
        )


class CompiledCallable:
    """A compiled wrapper around a Lucid model or function.

    Parameters
    ----------
    fn:
        The callable to compile.  Typically an :class:`~lucid.nn.Module` or
        a plain Python function that operates on :class:`~lucid.Tensor` objects.
    mode:
        Optimisation mode.  One of ``"default"``, ``"reduce-overhead"``, or
        ``"max-autotune"``.
    dynamic:
        When ``True``, shape changes trigger re-compilation silently.
        When ``False`` (default), a :class:`ShapeMismatchError` is raised.
    fullgraph:
        Reserved for future use.  When ``True``, the compiler will error out
        if it cannot capture the *entire* graph (no graph breaks).  Currently
        treated as a hint only.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        mode: str = "default",
        dynamic: bool = False,
        fullgraph: bool = False,
    ) -> None:
        self._fn = fn
        self.mode = mode
        self.dynamic = dynamic
        self.fullgraph = fullgraph

        self._cache: dict[tuple, CompilationResult] = {}
        self._call_count: int = 0
        self._hit_count: int = 0

    @property
    def fn(self) -> Callable:
        """The wrapped callable."""
        return self._fn

    def _make_cache_key(self, args: tuple, kwargs: dict) -> tuple:
        """Build a shape+dtype tuple key from tensor arguments."""
        parts: list[tuple] = []
        for arg in args:
            if isinstance(arg, lucid.Tensor):
                parts.append((tuple(arg.shape), arg.dtype, arg.device))
        for v in kwargs.values():
            if isinstance(v, lucid.Tensor):
                parts.append((tuple(v.shape), v.dtype, v.device))
        return tuple(parts)

    def _collect_input_tensors(
        self, args: tuple, kwargs: dict
    ) -> list[lucid.Tensor]:
        tensors: list[lucid.Tensor] = []
        for arg in args:
            if isinstance(arg, lucid.Tensor):
                tensors.append(arg)
        for v in kwargs.values():
            if isinstance(v, lucid.Tensor):
                tensors.append(v)
        return tensors

    def _compile(self, args: tuple, kwargs: dict) -> CompilationResult:
        """Trace the callable and apply optimisation passes."""
        t0 = time.perf_counter()

        input_tensors = self._collect_input_tensors(args, kwargs)

        with tracing_mode(input_tensors) as ctx:
            result = self._fn(*args, **kwargs)

        if isinstance(result, tuple):
            ctx.set_outputs(list(result))
        else:
            ctx.set_outputs([result])

        graph = ctx.graph

        if self.mode == "max-autotune":
            graph = run_max_passes(graph)
        else:
            graph = run_default_passes(graph)

        compile_time_ms = (time.perf_counter() - t0) * 1000.0
        key = self._make_cache_key(args, kwargs)

        return CompilationResult(
            graph=graph,
            input_key=key,
            compile_time_ms=compile_time_ms,
            mode=self.mode,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the compiled callable.

        On the first call with a given input-shape signature the forward pass
        is traced and the resulting graph is optimised.  Subsequent calls with
        the same signature use the cached graph and skip re-compilation.

        The actual tensor computation is always delegated to the original
        callable so that gradients flow correctly.
        """
        self._call_count += 1

        key = self._make_cache_key(args, kwargs)

        if key not in self._cache:
            if self._cache and not self.dynamic:
                existing = next(iter(self._cache))
                raise ShapeMismatchError(
                    f"Input shapes changed from {existing} to {key}. "
                    "Pass dynamic=True to allow recompilation."
                )
            self._cache[key] = self._compile(args, kwargs)
        else:
            self._hit_count += 1

        return self._fn(*args, **kwargs)

    def get_compilation_result(
        self, *args: Any, **kwargs: Any
    ) -> CompilationResult | None:
        """Return the cached :class:`CompilationResult` for given args, or ``None``."""
        key = self._make_cache_key(args, kwargs)
        return self._cache.get(key)

    def clear_cache(self) -> None:
        """Discard all cached compilation results."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of shape-specialised graphs in the cache."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        """Runtime statistics for diagnostics."""
        return {
            "call_count": self._call_count,
            "cache_hits": self._hit_count,
            "cache_misses": self._call_count - self._hit_count,
            "cache_size": self.cache_size,
        }

    def __repr__(self) -> str:
        fn_name = getattr(self._fn, "__name__", repr(self._fn))
        return (
            f"CompiledCallable("
            f"fn={fn_name!r}, "
            f"mode={self.mode!r}, "
            f"dynamic={self.dynamic}, "
            f"cache_size={self.cache_size})"
        )
