"""
lucid.compile — graph-capture compile path (3.5 Phase 1.1 Week 1 scaffold).

Phase 1.1 Week 1 only exposes :func:`_tracing` — a context manager that
installs a thread-local :class:`Tracer` so every op dispatched inside
the ``with`` block enters its :class:`OpScopeFull` and forwards a
header (name + single output meta) into the active recorder.

The user-facing decorator / module wrapper (:func:`compile`) is left as
a :class:`NotImplementedError` stub until Phase 1.4 lands the
:class:`CompiledModule` runtime; the MPSGraph builder + executable
cache land in Phase 1.2.  See ``~/.claude/plans/recursive-scribbling-moore.md``
for the full 6-phase plan.
"""

import sys
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    # The Tracer class lives in the `_C.engine.compile` sub-module.  The
    # bare attribute reference at runtime works under Python 3.14's lazy
    # annotation evaluation (PEP 649); we only need the import for the
    # static type checker.
    from lucid._C.engine.compile import Tracer
    from lucid.compile._compiled_module import CompiledModule
    from lucid.nn.module import Module

__all__ = [
    "_tracing",
    "compile",
    "compiled_step",
    "compile_optimizer",
    "fused_step",
    "CompiledModule",
    "make_step",
]


def __getattr__(name: str) -> object:
    # Lazy-load attributes that pull ``lucid.nn`` / ``lucid.autograd``
    # to avoid forcing those imports during ``lucid.compile`` package
    # init (the tracer scaffold above must stay importable before
    # nn/autograd have built their own surfaces).
    if name == "compiled_step":
        from lucid.compile._function import compiled_step

        return compiled_step
    if name == "CompiledModule":
        from lucid.compile._compiled_module import CompiledModule

        return CompiledModule
    if name == "make_step":
        from lucid.compile._step import make_step

        return make_step
    if name == "compile_optimizer":
        from lucid.compile._optim import compile_optimizer

        return compile_optimizer
    if name == "fused_step":
        from lucid.compile._fused_step import fused_step

        return fused_step
    raise AttributeError(f"module 'lucid.compile' has no attribute {name!r}")


@contextmanager
def _tracing() -> Iterator[Tracer]:
    """Install a fresh :class:`Tracer` for the calling thread.

    Every op dispatched inside the ``with`` block enters its
    :class:`OpScopeFull` constructor and forwards its header (name +
    output meta) into the active tracer.  On exit the tracer is detached
    and its recorded :class:`TraceGraph` becomes a frozen view that can
    still be inspected via ``tracer.graph``.

    Yields
    ------
    Tracer
        The freshly-installed tracer.  Useful both for inspection during
        the trace (``len(tracer.graph)`` grows monotonically) and after
        (the final recorded DAG).

    Notes
    -----
    Phase 1.1 only records op headers — input-id wiring and MPSGraph
    emission arrive in Phase 1.2.  The context manager today is the
    smoke-test surface that proves the :class:`OpScopeFull` hook is
    correctly integrated; it is intentionally a leading underscore name
    so callers do not depend on the partial Week-1 surface.

    Examples
    --------
    Smoke-test pattern::

        import lucid
        import lucid.nn as nn

        model = nn.Linear(8, 4)
        x = lucid.zeros((2, 8))

        with lucid.compile._tracing() as tracer:
            _ = model(x)

        for node in tracer.graph.ops:
            print(node)
    """
    tracer = _C_engine.compile.Tracer()
    _C_engine.compile.set_current_tracer(tracer)
    try:
        yield tracer
    finally:
        _C_engine.compile.set_current_tracer(None)
        # LUCID_COMPILE_DEBUG=1 writes the captured TraceGraph to a
        # temp-dir JSON file for offline inspection (Phase 1.2 builder
        # + parity tooling will consume the same format).  No-op when
        # the env var is unset.
        from lucid.compile._trace_dump import dump_to_path_if_debug_enabled

        dump_to_path_if_debug_enabled(tracer.graph)


def compile(target: object, *, dynamic: bool = False) -> object:
    """Wrap ``target`` so calls are routed through cached MPSGraph executables.

    Accepts three call styles:

    * **Module wrapping** — ``lucid.compile(model)`` returns a
      :class:`CompiledModule` that delegates parameter walks /
      ``state_dict`` / device moves to ``model`` while routing
      ``__call__`` through an executable cache.
    * **Plain callable wrapping** — ``lucid.compile(fn)`` where ``fn``
      is a regular function or any callable.  Returns a thin wrapper
      whose ``__call__`` traces the function's body on the first
      invocation with a new input signature and caches an executable.
      The wrapper has no parameters of its own — pure tensor-in /
      tensor-out.
    * **Decorator usage** — ``@lucid.compile`` on a function (or
      ``@lucid.compile(dynamic=True)`` factory form) — identical to
      the plain-callable form above but lets the user opt-in inline.

    Parameters
    ----------
    target : nn.Module or callable
        Either an :class:`nn.Module` instance (preferred when the
        compiled unit carries learnable parameters) or any callable
        whose signature is ``(*tensor_args) -> Tensor | tuple | dict``.
    dynamic : bool, optional
        Opt-in to symbolic batch-dim shape support (Phase 1.6).  Today
        only the static path is implemented — passing ``True`` raises
        ``NotImplementedError``.

    Returns
    -------
    CompiledModule
        A wrapper exposing the cache + run path.  When ``target`` was a
        Module the wrapper re-exposes the inner module's
        ``parameters`` / ``state_dict`` / training mode; when it was a
        plain callable those return empty sequences.

    Examples
    --------
    Module form::

        compiled = lucid.compile(nn.Linear(8, 4).to('metal'))
        y = compiled(x)

    Callable form::

        @lucid.compile
        def f(x, y):
            return (x @ y.T).relu()

        z = f(a, b)

    Factory decorator with kwargs::

        @lucid.compile(dynamic=False)
        def attention(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)
    """

    # Factory form: ``@lucid.compile(dynamic=...)`` with no positional —
    # ``target`` is the callable that comes back via the returned
    # decorator.  Detect by ``target`` being missing on the call site;
    # the module ``__call__`` below short-circuits the explicit form.
    from lucid.compile._compiled_module import CompiledModule as _CM
    from lucid.nn.module import Module as _Module

    if isinstance(target, _Module):
        return _CM(target, dynamic=dynamic)
    if callable(target):
        # Wrap plain callable as a synthetic Module so the existing
        # CompiledModule machinery (signature_of / trace / cache) works
        # without changes.  The synthetic module holds zero parameters
        # so ``param_fingerprint`` collapses to an empty tuple.
        return _CM(_CallableModule(target), dynamic=dynamic)
    raise TypeError(
        f"lucid.compile: target must be an nn.Module or a callable, "
        f"got {type(target).__name__}"
    )


class _CallableModule:
    """Internal adapter: wraps a plain callable as a minimal Module-like.

    Provides only the slots :class:`CompiledModule` consults
    (``training``, ``parameters``, ``state_dict``, ``to``, ``train``,
    ``__call__``).  Has zero parameters; never registers anything in
    ``_modules``.  ``__getattr__`` raises so any missing attribute is
    surfaced loudly rather than silently passed through.
    """

    def __init__(self, fn: object) -> None:
        # Avoid Module.__setattr__ side-effects — we are *not* an
        # nn.Module subclass, just duck-typed.
        self._fn = fn
        self._training: bool = False

    # ── Module-shaped surface ────────────────────────────────────
    @property
    def training(self) -> bool:
        return self._training

    def train(self, mode: bool = True) -> "_CallableModule":
        self._training = bool(mode)
        return self

    def eval(self) -> "_CallableModule":
        return self.train(False)

    def parameters(self, recurse: bool = True) -> "Iterator[object]":
        return iter(())

    def named_parameters(self, recurse: bool = True) -> "Iterator[tuple[str, object]]":
        return iter(())

    def buffers(self, recurse: bool = True) -> "Iterator[object]":
        return iter(())

    def state_dict(self, *args: object, **kwargs: object) -> dict:
        return {}

    def load_state_dict(self, *args: object, **kwargs: object) -> None:
        return None

    def to(self, *args: object, **kwargs: object) -> "_CallableModule":
        # No-op — plain callable carries no buffers to move.
        return self

    def modules(self) -> "Iterator[object]":
        return iter((self,))

    def named_modules(
        self, *args: object, **kwargs: object
    ) -> "Iterator[tuple[str, object]]":
        return iter((("", self),))

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._fn(*args, **kwargs)

    # Allow CompiledModule's signature_of (``model.training`` /
    # ``model.parameters``) to work without surprises.
    def __repr__(self) -> str:
        name = getattr(self._fn, "__qualname__", repr(self._fn))
        return f"<_CallableModule wrapping {name}>"


def _compile_decorator_factory(*, dynamic: bool = False) -> "object":
    """Return a decorator that applies :func:`compile` with the given options.

    Powers the ``@lucid.compile(dynamic=False)`` factory form when the
    user calls the module-level wrapper with only kwargs and no
    positional target.
    """

    def _decorator(target: object) -> object:
        return compile(target, dynamic=dynamic)

    return _decorator


class _CallableCompileModule(types.ModuleType):
    """Module subclass so ``lucid.compile(...)`` is callable.

    Handles three signatures:

    1. ``lucid.compile(module_or_callable)`` → :func:`compile`.
    2. ``lucid.compile(module_or_callable, dynamic=False)`` → same.
    3. ``lucid.compile(dynamic=False)`` (no positional) → returns a
       decorator factory so ``@lucid.compile(dynamic=False)`` works.
    """

    def __call__(self, *args: object, **kwargs: object) -> object:
        if not args:
            # Factory form: ``@lucid.compile(dynamic=...)``.
            return _compile_decorator_factory(**kwargs)
        if len(args) != 1:
            raise TypeError(
                f"lucid.compile: expected 0 or 1 positional argument, "
                f"got {len(args)}"
            )
        return compile(args[0], **kwargs)


sys.modules[__name__].__class__ = _CallableCompileModule
