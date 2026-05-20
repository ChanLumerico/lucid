"""Allocation-free model instantiation context for the docs/summary
pipeline.

Background
----------
Computing a paper-faithful layer summary tree for a model normally
requires actually constructing the model — the only reliable way to
walk ``named_children()`` and read ``Parameter.shape`` is to run each
sub-module's ``__init__``.  That costs **real memory**: ViT-Huge needs
~2 GB, GPT-2-XL ~6 GB, and anything in the hundreds-of-billions
regime is simply unreachable on a developer laptop or a hosted CI
runner.

This module provides a context manager :func:`shadow_alloc` that:

  - Hot-patches the engine's tensor-creation entry points
    (``_C_engine.zeros / ones / empty / full / eye / arange / randn /
    reshape``) to return phantom impls carrying **only shape /
    dtype / device** metadata.
  - Leaves every Python-level Module / Parameter / init helper
    untouched.  Their ``__init__`` runs end-to-end, but the
    ``_impl`` they end up holding is a phantom — no Storage, no
    MLX array, no zero-init kernel.

For the docs use case (``compute_model_summary``) this is sufficient:
the summary walker only reads ``p.shape`` per parameter and the
parent/child structure via ``named_children()``.  Both work on
phantom impls because the structure is set at the Python level.

Constraints
-----------
- This context is **only safe for object construction**.  Any code
  path that *runs* a model (``forward``, ``backward``, op kernels)
  will fault — phantom impls don't support actual compute.
- Random initialization is silently skipped — fine for the structural
  summary, but never use a shadow-constructed model for training.
"""

import contextlib
from typing import Any, Iterator

from lucid._C import engine as _C_engine

_SHADOW_ENABLED: bool = False


# ---------------------------------------------------------------------------
# Phantom impl
# ---------------------------------------------------------------------------


class PhantomImpl:
    """Stand-in for :class:`_C_engine.TensorImpl` in shadow mode.

    Carries shape / dtype / device metadata and absorbs every other
    method call as a no-op so the existing Module / init code paths
    don't need any special-casing.
    """

    __slots__ = ("shape", "dtype", "device", "requires_grad", "ndim")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: Any = None,
        device: Any = None,
        requires_grad: bool = False,
    ) -> None:
        # Normalise shape to a flat tuple of ints — incoming shapes can
        # be int, list, tuple, or even another phantom's .shape.
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.ndim = len(self.shape)

    # Methods that lucid code paths *do* call on impls during __init__:

    def clone_with_grad(self, requires_grad: bool) -> PhantomImpl:
        return PhantomImpl(self.shape, self.dtype, self.device, requires_grad)

    def reshape(self, shape) -> PhantomImpl:
        return PhantomImpl(shape, self.dtype, self.device, self.requires_grad)

    def detach(self) -> PhantomImpl:
        return self

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    # Universal absorber for any other engine method invoked on an impl
    # (init helpers in nn/init.py call lots of these).  Returns a
    # chainable lambda so ``impl.foo(...).bar(...)`` style fluent calls
    # don't blow up.
    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _PhantomMethod(self)


class _PhantomMethod:
    """Callable proxy for unknown methods — returns the phantom impl
    so ``impl.unknown_op(...).another(...)`` keeps chaining."""

    __slots__ = ("_impl",)

    def __init__(self, impl: PhantomImpl) -> None:
        self._impl = impl

    def __call__(self, *args: Any, **kwargs: Any) -> PhantomImpl:
        return self._impl


# ---------------------------------------------------------------------------
# Patched engine factories
# ---------------------------------------------------------------------------


def _shape_only_factory(*args: Any, **kwargs: Any) -> PhantomImpl:
    """Engine signature: ``f(shape, dtype, device)`` — zeros / ones /
    empty / eye all match this layout."""
    if not args:
        return PhantomImpl((), None, None)
    shape = args[0]
    dtype = args[1] if len(args) >= 2 else kwargs.get("dtype")
    device = args[2] if len(args) >= 3 else kwargs.get("device")
    return PhantomImpl(shape, dtype, device)


def _full_factory(shape: Any, value: Any, *args: Any, **kwargs: Any) -> PhantomImpl:
    """Engine signature: ``full(shape, value, dtype, device)``."""
    dtype = args[0] if len(args) >= 1 else kwargs.get("dtype")
    device = args[1] if len(args) >= 2 else kwargs.get("device")
    return PhantomImpl(shape, dtype, device)


def _arange_factory(
    start: float, end: float | None = None, step: float = 1, *args: Any, **kwargs: Any
) -> PhantomImpl:
    """Engine signature: ``arange(start, end, step, dtype, device)`` —
    we still derive a length for accurate shape reporting."""
    if end is None:
        start, end = 0, start
    length = max(
        0, int((float(end) - float(start) + step - (1 if step > 0 else -1)) // step)
    )
    dtype = args[2] if len(args) >= 3 else kwargs.get("dtype")
    device = args[3] if len(args) >= 4 else kwargs.get("device")
    return PhantomImpl((length,), dtype, device)


def _reshape_factory(impl: Any, shape: Any, *args: Any, **kwargs: Any) -> PhantomImpl:
    """``reshape(impl, shape)`` — used by ``init._fill_from_impl``.
    Pass-through for phantom input; defer to real engine otherwise."""
    if isinstance(impl, PhantomImpl):
        return PhantomImpl(shape, impl.dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["reshape"](impl, shape, *args, **kwargs)


def _find_phantom(args: tuple[Any, ...]) -> PhantomImpl | None:
    """Return the first phantom found anywhere in ``args`` — at the top
    level *or* inside a directly-nested list/tuple (e.g. the first
    element of a ``concatenate([t1, t2, ...], dim)`` call).  One level
    of unwrapping is enough; deeper nesting isn't used by engine ops."""
    for a in args:
        if isinstance(a, PhantomImpl):
            return a
        if isinstance(a, (list, tuple)):
            for inner in a:
                if isinstance(inner, PhantomImpl):
                    return inner
    return None


def _shape_preserving_wrap(orig_name: str, orig: Any) -> Any:
    """Default phantom-aware wrapper: if any arg (including elements of
    a top-level list/tuple) is a phantom, return a phantom with that
    arg's shape.  Falls through to the real engine when nothing
    phantom is present so non-shadow paths in the same process aren't
    affected.

    The shape isn't exact for shape-changing ops (cat, stack, slice),
    but the only consumers in module ``__init__`` paths are
    ``init._fill_from_impl`` (which immediately reshapes to the
    Parameter's true shape) and the no-op chain inside
    ``PhantomImpl.__getattr__``, so the inaccuracy is harmless.
    """

    def w(*args: Any, **kwargs: Any) -> Any:
        ph = _find_phantom(args)
        if ph is not None:
            return PhantomImpl(ph.shape, ph.dtype, ph.device, ph.requires_grad)
        return orig(*args, **kwargs)

    return w


def _explicit_shape_wrap(orig_name: str, orig: Any) -> Any:
    """For ops whose output shape = **second positional arg** (the
    shape parameter): ``reshape(t, shape)``, ``broadcast_to(t, shape)``,
    ``expand(t, shape)``, etc."""

    def w(impl: Any, shape: Any, *rest: Any, **kwargs: Any) -> Any:
        if isinstance(impl, PhantomImpl):
            return PhantomImpl(shape, impl.dtype, impl.device, impl.requires_grad)
        return orig(impl, shape, *rest, **kwargs)

    return w


# Engine ops whose output shape is the *explicit* shape argument (not
# derived from input).  Anything not in this list defaults to
# shape-preserving (output shape == first phantom-arg's shape).
_EXPLICIT_SHAPE_OPS = frozenset(
    {
        "reshape",
        "view",
        "broadcast_to",
        "expand",
    }
)


# Engine functions intercepted while shadow_alloc is active.  Targeted
# list — we only patch what lucid Modules / init helpers exercise during
# object construction.

# Engine functions whose semantics need a custom wrapper (not the
# generic shape-preserving default).  Anything not listed here is
# auto-wrapped to "first phantom arg's shape pass-through".
_SPECIFIC_WRAPPERS: dict[str, Any] = {
    # Pure tensor factories — ``f(shape, ..., dtype, device)``.  These
    # take a shape as the first positional arg and create a brand-new
    # tensor; no phantom input means the generic shape-preserving
    # wrapper would miss them.
    "zeros": _shape_only_factory,
    "ones": _shape_only_factory,
    "empty": _shape_only_factory,
    "eye": _shape_only_factory,
    "randn": _shape_only_factory,
    "rand": _shape_only_factory,
    "uniform": _shape_only_factory,
    "normal": _shape_only_factory,
    "bernoulli": _shape_only_factory,
    "exponential": _shape_only_factory,
    "full": _full_factory,
    "arange": _arange_factory,
    "reshape": _reshape_factory,
}

_ORIGINAL: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------


def is_active() -> bool:
    """Return ``True`` while inside a ``shadow_alloc`` block."""
    return _SHADOW_ENABLED


@contextlib.contextmanager
def shadow_alloc() -> Iterator[None]:
    r"""Construct models without allocating any tensor storage.

    Use as a context manager around factory invocations whose only
    purpose is to inspect structure (e.g. layer-summary generation
    for the docs site).  Inside the block the engine's tensor-creation
    entry points return phantom impls; no real memory is reserved
    even for hundreds-of-billions-parameter models.

    Examples
    --------
    >>> from lucid.nn._shadow import shadow_alloc
    >>> from lucid.models.vision.resnet import resnet_152
    >>> with shadow_alloc():
    ...     model = resnet_152()  # constructs without GB of RAM
    >>> # Only structural metadata is meaningful afterwards:
    >>> sum(p.shape.numel() if hasattr(p.shape, 'numel') else 1
    ...     for p in model.parameters())  # works
    """
    global _SHADOW_ENABLED
    if _SHADOW_ENABLED:
        yield
        return

    # Patch ``_unwrap`` to accept phantom impls.  High-level op dispatch
    # (``lucid.where``, comparison ops, …) routes every Tensor through
    # ``_unwrap`` before calling the engine; the default impl rejects
    # anything that isn't an actual ``_C_engine.TensorImpl``.  In shadow
    # mode we extend the gate to also pass phantoms through.  Note:
    # ``_unwrap`` was imported as a name into ``lucid._ops`` and
    # ``lucid._ops._adapters`` at module-load time, so simply rebinding
    # ``_disp._unwrap`` isn't enough — we also rebind the already-
    # captured references in those modules.
    from lucid import _dispatch as _disp
    from lucid import _ops as _ops_pkg
    from lucid._ops import _adapters as _adapters_mod

    _orig_unwrap_disp = _disp._unwrap
    _orig_unwrap_ops = _ops_pkg._unwrap
    _orig_unwrap_adap = _adapters_mod._unwrap

    def _shadow_unwrap(t: Any) -> Any:
        impl = getattr(t, "_impl", None)
        if isinstance(impl, PhantomImpl):
            return impl
        return _orig_unwrap_disp(t)

    _disp._unwrap = _shadow_unwrap
    _ops_pkg._unwrap = _shadow_unwrap
    _adapters_mod._unwrap = _shadow_unwrap
    _ORIGINAL["__unwrap_disp__"] = _orig_unwrap_disp
    _ORIGINAL["__unwrap_ops__"] = _orig_unwrap_ops
    _ORIGINAL["__unwrap_adap__"] = _orig_unwrap_adap

    # Snapshot + patch every callable attribute on the engine module.
    # Tensor-creating ops with shape semantics that differ from "pass
    # through the first phantom's shape" get a custom wrapper from
    # ``_SPECIFIC_WRAPPERS``; ops whose output shape is the explicit
    # ``shape`` argument (``reshape``, ``broadcast_to`` ...) use
    # ``_explicit_shape_wrap``; everything else gets the generic
    # phantom-aware fallback.  Classes and constants (TensorImpl,
    # Device, Dtype enums, ...) are left untouched.
    for name in dir(_C_engine):
        if name.startswith("_"):
            continue
        attr = getattr(_C_engine, name)
        if not callable(attr) or isinstance(attr, type):
            continue
        _ORIGINAL[name] = attr
        if name in _SPECIFIC_WRAPPERS:
            setattr(_C_engine, name, _SPECIFIC_WRAPPERS[name])
        elif name in _EXPLICIT_SHAPE_OPS:
            setattr(_C_engine, name, _explicit_shape_wrap(name, attr))
        else:
            setattr(_C_engine, name, _shape_preserving_wrap(name, attr))

    # ``_ops/_registry.OpEntry`` instances captured ``_C_engine.<op>``
    # references at import time, so re-bind each entry's ``engine_fn``
    # to the freshly-patched attribute looked up by name.  Tensor methods
    # (``a.unsqueeze(0)``) and ``lucid.<op>`` free functions both dispatch
    # through these entries, so without this step they'd still hit the
    # original C++ binding and reject phantom inputs.
    from lucid._ops._registry import _REGISTRY as _OP_REGISTRY

    _op_originals: list[tuple[Any, Any]] = []
    for _e in _OP_REGISTRY:
        new_fn = getattr(_C_engine, _e.name, None)
        if new_fn is not None and new_fn is not _e.engine_fn:
            _op_originals.append((_e, _e.engine_fn))
            _e.engine_fn = new_fn
    _ORIGINAL["__op_registry__"] = _op_originals

    _SHADOW_ENABLED = True
    try:
        yield
    finally:
        _SHADOW_ENABLED = False
        # Restore dispatch unwraps first (sentinel keys — not on engine).
        if "__unwrap_disp__" in _ORIGINAL:
            _disp._unwrap = _ORIGINAL.pop("__unwrap_disp__")
        if "__unwrap_ops__" in _ORIGINAL:
            _ops_pkg._unwrap = _ORIGINAL.pop("__unwrap_ops__")
        if "__unwrap_adap__" in _ORIGINAL:
            _adapters_mod._unwrap = _ORIGINAL.pop("__unwrap_adap__")
        # Roll back the OpEntry.engine_fn rebindings.
        op_pairs = _ORIGINAL.pop("__op_registry__", [])
        for _e, _orig_fn in op_pairs:
            _e.engine_fn = _orig_fn
        for name, orig in _ORIGINAL.items():
            setattr(_C_engine, name, orig)
        _ORIGINAL.clear()
