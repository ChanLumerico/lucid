"""
lucid.compile
-------------
Graph compilation for Lucid models and functions.

Quick start
~~~~~~~~~~~
::

    import lucid
    import lucid.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            return self.fc2(lucid.relu(self.fc1(x)))

    model = MLP()

    # --- Option A: top-level function ---
    compiled = lucid.compile(model)
    y = compiled(x)

    # --- Option B: Module method ---
    compiled = model.compile()
    y = compiled(x)

    # --- Option C: decorator ---
    @lucid.compile
    def my_fn(x):
        return model(x)

    y = my_fn(x)

Compile modes
~~~~~~~~~~~~~
``"default"``
    Dead-code elimination + operator fusion.  Always safe, works in both
    training and inference.

``"reduce-overhead"``
    Same graph passes as ``"default"`` but skips redundant Python-level
    dispatch on cache hits, lowering per-call overhead.

``"max-autotune"``
    Adds constant folding on top of the default passes.  Best for
    inference-only workloads with frozen weights.  Do **not** use during
    training – folded constants will become stale if weights change.

Dynamic shapes
~~~~~~~~~~~~~~
By default ``compile()`` raises :class:`~lucid.compile.ShapeMismatchError`
when input shapes change between calls.  Pass ``dynamic=True`` to allow
silent recompilation for each new shape signature.

Introspection
~~~~~~~~~~~~~
::

    result = compiled.get_compilation_result(x)
    print(result.graph.summary())   # human-readable IR listing
    print(result.compile_time_ms)   # wall-clock ms for the trace + passes
    print(compiled.stats)           # call count, cache hits/misses

Custom fusion rules
~~~~~~~~~~~~~~~~~~~
::

    from lucid.compile import register_fusion_rule, FusionRule
    register_fusion_rule(FusionRule("my_op", "relu", "fused:my_op_relu"))
"""

from __future__ import annotations

from typing import Callable, Any, overload

from lucid.compile._ir import IRGraph, IRNode
from lucid.compile._trace import TraceContext, tracing_mode, get_trace_context, is_tracing
from lucid.compile._compiled import CompiledCallable, CompilationResult, ShapeMismatchError
from lucid.compile._passes import (
    dead_code_elimination,
    constant_folding,
    operator_fusion,
    register_fusion_rule,
    run_default_passes,
    run_max_passes,
)
from lucid.compile._passes.operator_fusion import FusionRule

__all__ = [
    "compile",
    "IRGraph",
    "IRNode",
    "TraceContext",
    "tracing_mode",
    "get_trace_context",
    "is_tracing",
    "CompiledCallable",
    "CompilationResult",
    "ShapeMismatchError",
    "dead_code_elimination",
    "constant_folding",
    "operator_fusion",
    "register_fusion_rule",
    "run_default_passes",
    "run_max_passes",
    "FusionRule",
]

_VALID_MODES = {"default", "reduce-overhead", "max-autotune"}


@overload
def compile(fn: Callable) -> CompiledCallable: ...


@overload
def compile(
    fn: Callable,
    *,
    mode: str,
    dynamic: bool,
    fullgraph: bool,
) -> CompiledCallable: ...


def compile(
    fn: Callable | None = None,
    *,
    mode: str = "default",
    dynamic: bool = False,
    fullgraph: bool = False,
) -> "CompiledCallable | Callable[..., CompiledCallable]":
    """Compile *fn* for faster repeated execution.

    Can be used as a plain function call, a keyword-argument factory, or a
    decorator (with or without arguments).

    Parameters
    ----------
    fn:
        The callable to compile – an :class:`~lucid.nn.Module` or a plain
        function that takes and returns :class:`~lucid.Tensor` objects.
        When omitted the function returns a *decorator factory* so that
        ``@lucid.compile(mode="max-autotune")`` syntax works.
    mode:
        Optimisation level.  One of ``"default"``, ``"reduce-overhead"``,
        or ``"max-autotune"``.
    dynamic:
        Allow silent recompilation when input shapes change (default:
        ``False``).
    fullgraph:
        Raise on graph breaks (not yet enforced; reserved for future use).

    Returns
    -------
    CompiledCallable
        When *fn* is provided.
    Callable → CompiledCallable
        A decorator factory when *fn* is ``None``.

    Examples
    --------
    ::

        # Plain call
        fast_model = lucid.compile(model)

        # Decorator without arguments
        @lucid.compile
        def forward(x):
            return model(x)

        # Decorator with arguments
        @lucid.compile(mode="max-autotune", dynamic=True)
        def inference(x):
            return model(x)
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown compile mode {mode!r}. "
            f"Valid modes: {sorted(_VALID_MODES)}"
        )

    def _make(target: Callable) -> CompiledCallable:
        return CompiledCallable(
            target,
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )

    if fn is not None:
        return _make(fn)

    return _make
