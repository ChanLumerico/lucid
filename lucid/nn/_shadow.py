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
        # Default dtype/device to F32/CPU when None — empty values trip
        # ``_maybe_promote`` (it then calls ``astype(impl, None)`` which
        # the engine binding rejects).  RoFormer / CoAtNet hit this path
        # during ``__init__`` via arithmetic on freshly-created
        # ``arange`` phantoms.
        self.dtype = dtype if dtype is not None else _C_engine.F32
        self.device = device if device is not None else _C_engine.CPU
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

    # ── Indexing ────────────────────────────────────────────────────────────
    # Swin / Mask2Former call ``relative_position_bias_table[index]`` on a
    # phantom Parameter during ``__init__`` to materialise a lookup
    # table.  Returning a same-shape phantom is best-effort but enough
    # for the layer-summary walker — the shape downstream code reads off
    # the indexed result is irrelevant when no actual values flow.
    def __getitem__(self, key: Any) -> PhantomImpl:
        return PhantomImpl(self.shape, self.dtype, self.device, self.requires_grad)

    def __setitem__(self, key: Any, value: Any) -> None:
        # In-place assignment — no-op in shadow mode.
        return None

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


def _meshgrid_factory(tensors: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """``meshgrid([t1, t2, ...], indexing=...)`` returns **N tensors**
    (one per input) — generic shape-preserving wrapper would collapse
    that to a single phantom and break the caller's tuple unpack
    (``gy, gx = lucid.meshgrid(a, b)`` in Swin's window attention).
    Output shape per tensor: combined dim sizes of all inputs.
    """
    if isinstance(tensors, (list, tuple)) and any(
        isinstance(t, PhantomImpl) for t in tensors
    ):
        # Combined shape = product of input lengths
        combined = tuple(
            t.shape[0] if isinstance(t, PhantomImpl) and t.shape else 1 for t in tensors
        )
        # Pick a representative phantom for dtype/device
        rep = next(t for t in tensors if isinstance(t, PhantomImpl))
        return [PhantomImpl(combined, rep.dtype, rep.device, False) for _ in tensors]
    return _ORIGINAL["meshgrid"](tensors, *args, **kwargs)


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
    "meshgrid": _meshgrid_factory,
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

    # Patch ``_unwrap`` to accept phantom impls.  ~20+ modules across
    # ``lucid/`` do ``from lucid._dispatch import _unwrap`` at
    # module-load time, so we can't just rebind ``_dispatch._unwrap``
    # — each importer captured the function object then.  Walk
    # ``sys.modules`` once and replace every module-level ``_unwrap``
    # attribute that *is* the original (identity check).
    import sys
    from lucid import _dispatch as _disp

    _orig_unwrap = _disp._unwrap

    def _shadow_unwrap(t: Any) -> Any:
        impl = getattr(t, "_impl", None)
        if isinstance(impl, PhantomImpl):
            return impl
        return _orig_unwrap(t)

    _unwrap_rebindings: list[Any] = []
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        if getattr(_mod, "_unwrap", None) is _orig_unwrap:
            _mod._unwrap = _shadow_unwrap
            _unwrap_rebindings.append(_mod)
    _ORIGINAL["__unwrap_sites__"] = (_orig_unwrap, _unwrap_rebindings)

    # Same treatment for ``_unwrap_or_scalar`` — defined in
    # ``lucid._tensor._dunders`` and used by every binary-arithmetic
    # dunder.  CoAtNet's ``_init_rel_idx`` exercises this path during
    # ``__init__`` (computing relative-position offsets on tiny
    # arange-backed tensors).  The default impl rejects anything whose
    # impl isn't a TensorImpl; we extend it to also accept phantoms.
    from lucid._tensor import _dunders as _dunders_mod

    _orig_uos = _dunders_mod._unwrap_or_scalar

    def _shadow_uos(x: Any, ref_impl: Any = None) -> Any:
        impl = getattr(x, "_impl", None)
        if isinstance(impl, PhantomImpl):
            return impl
        if isinstance(x, PhantomImpl):
            return x
        return _orig_uos(x, ref_impl)

    _uos_rebindings: list[Any] = []
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        if getattr(_mod, "_unwrap_or_scalar", None) is _orig_uos:
            _mod._unwrap_or_scalar = _shadow_uos
            _uos_rebindings.append(_mod)
    _ORIGINAL["__uos_sites__"] = (_orig_uos, _uos_rebindings)

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

    # IMPORTANT: only rebind when ``engine_fn`` IS the (pre-patch)
    # engine attribute — entries whose engine_fn is a Python *adapter*
    # (e.g. ``_where_adapter``) handle their own ``_unwrap`` /
    # pre-processing internally and break if blindly replaced with the
    # engine-level patch.  Adapters already call into ``_C_engine.<op>``
    # which we *did* patch, so phantom routing still happens via them.
    _op_originals: list[tuple[Any, Any]] = []
    for _e in _OP_REGISTRY:
        orig_engine_attr = _ORIGINAL.get(_e.name)
        if orig_engine_attr is None:
            continue
        if _e.engine_fn is not orig_engine_attr:
            continue  # adapter or already-wrapped — leave alone
        new_fn = getattr(_C_engine, _e.name, None)
        if new_fn is None or new_fn is _e.engine_fn:
            continue
        _op_originals.append((_e, _e.engine_fn))
        _e.engine_fn = new_fn
    _ORIGINAL["__op_registry__"] = _op_originals

    _SHADOW_ENABLED = True
    try:
        yield
    finally:
        _SHADOW_ENABLED = False
        # Restore the ``_unwrap`` rebindings.
        sites = _ORIGINAL.pop("__unwrap_sites__", None)
        if sites is not None:
            _orig_uw, _mods = sites
            for _mod in _mods:
                _mod._unwrap = _orig_uw
        # Restore the ``_unwrap_or_scalar`` rebindings.
        sites = _ORIGINAL.pop("__uos_sites__", None)
        if sites is not None:
            _orig_uw, _mods = sites
            for _mod in _mods:
                _mod._unwrap_or_scalar = _orig_uw
        # Roll back the OpEntry.engine_fn rebindings.
        op_pairs = _ORIGINAL.pop("__op_registry__", [])
        for _e, _orig_fn in op_pairs:
            _e.engine_fn = _orig_fn
        for name, orig in _ORIGINAL.items():
            setattr(_C_engine, name, orig)
        _ORIGINAL.clear()
