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
- The block belongs to the thread that opens it.  The patches are
  process-wide while any block is open, but each one checks the
  calling thread's state, so every other thread — one started inside
  the block included — keeps computing real values.
"""

import contextlib
import contextvars
import math
import sys
import threading
import types
from collections.abc import Callable, Iterator
from typing import Any, cast, final

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor

# Set for the duration of a block, in the context (thread) that opened it.
# A thread starts with a fresh context, so it never inherits a block.
_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "lucid_shadow_alloc_active", default=False
)


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
        # ``_promotion`` / ``_dispatch`` (e.g. ``astype(impl, None)`` which
        # the engine binding rejects).  RoFormer / CoAtNet hit this path
        # during ``__init__`` via arithmetic on freshly-created
        # ``arange`` phantoms.
        self.dtype = dtype if dtype is not None else _C_engine.F32
        self.device = device if device is not None else _C_engine.CPU
        self.requires_grad = requires_grad
        self.ndim = len(self.shape)

    @property
    def retains_grad(self) -> bool:
        return False

    def is_aliased(self) -> bool:
        return False

    def retain_grad_(self) -> None:
        return None

    # Methods that lucid code paths *do* call on impls during __init__:

    def clone_with_grad(self, requires_grad: bool) -> PhantomImpl:
        return PhantomImpl(self.shape, self.dtype, self.device, requires_grad)

    def data_alias(self) -> PhantomImpl:
        return PhantomImpl(self.shape, self.dtype, self.device, False)

    def reshape(self, shape: Any) -> PhantomImpl:
        return PhantomImpl(
            _resolve_reshape(self.shape, shape),
            self.dtype,
            self.device,
            self.requires_grad,
        )

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
    def __getattr__(self, name: str) -> _PhantomMethod:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _PhantomMethod(self)


@final
class _PhantomMethod:
    """Callable proxy for unknown methods — returns the phantom impl
    so ``impl.unknown_op(...).another(...)`` keeps chaining."""

    __slots__ = ("_impl",)

    def __init__(self, impl: PhantomImpl) -> None:
        self._impl = impl

    def __call__(self, *args: Any, **kwargs: Any) -> PhantomImpl:
        return self._impl


# ---------------------------------------------------------------------------
# Shape and dtype helpers for phantom ops
# ---------------------------------------------------------------------------


def _broadcast_shapes(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Broadcast two shape tuples following the standard right-aligned rule."""
    if a == b:
        return a
    if not a:
        return b
    if not b:
        return a
    rank = max(len(a), len(b))
    pa = (1,) * (rank - len(a)) + a
    pb = (1,) * (rank - len(b)) + b
    out: list[int] = []
    for x, y in zip(pa, pb, strict=True):
        if x != y and 1 not in (x, y):
            raise _C_engine.ShapeMismatch(
                f"Shapes {list(a)} and {list(b)} cannot be broadcast"
            )
        out.append(y if x == 1 else x)
    return tuple(out)


def _resolve_reshape(src_shape: tuple[int, ...], target: Any) -> tuple[int, ...]:
    """Resolve a reshape target (including up to one ``-1`` dim) against ``src_shape``."""
    raw = (int(target),) if isinstance(target, int) else tuple(int(s) for s in target)
    if -1 not in raw:
        return raw
    known = 1
    neg_idx = -1
    for i, s in enumerate(raw):
        if s == -1:
            neg_idx = i
        else:
            known *= s
    total = math.prod(src_shape)
    inferred = total // known if known > 0 else 0
    return (*raw[:neg_idx], inferred, *raw[neg_idx + 1 :])


def _promote_two(
    a: Any, b: Any, *, true_div: bool = False, opmath: bool = False
) -> Any:
    """Compute the promoted engine dtype for two impls via :mod:`lucid._promotion`."""
    from lucid import _promotion

    if opmath and a.dtype != b.dtype:
        if (
            _promotion.is_half(a.dtype)
            and not b.shape
            and b.dtype in (_C_engine.F32, _C_engine.F64)
        ):
            return a.dtype
        if (
            _promotion.is_half(b.dtype)
            and not a.shape
            and a.dtype in (_C_engine.F32, _C_engine.F64)
        ):
            return b.dtype
    dt = _promotion.result_type(a, b)
    if true_div and _promotion.is_integral(dt):
        dt = _promotion.default_float(a.device)
    return dt


def _reduced_shape(shape: tuple[int, ...], axes: Any, keepdim: bool) -> tuple[int, ...]:
    """Compute output shape for a reduction along ``axes``."""
    ndim = len(shape)
    if ndim == 0:
        return ()
    if axes is None:
        norm = set(range(ndim))
    elif isinstance(axes, int):
        norm = {axes % ndim}
    else:
        axis_list = [int(a) % ndim for a in axes]
        norm = set(axis_list) if axis_list else set(range(ndim))
    if keepdim:
        return tuple(1 if i in norm else s for i, s in enumerate(shape))
    return tuple(s for i, s in enumerate(shape) if i not in norm)


# ---------------------------------------------------------------------------
# Patched engine factories and ops
# ---------------------------------------------------------------------------


def _shape_only_factory(*args: Any, **kwargs: Any) -> PhantomImpl:
    """Engine signature: ``f(shape, dtype, device)`` — zeros / ones /
    empty / rand / randn all match this layout."""
    if not args:
        return PhantomImpl((), None, None)
    shape = args[0]
    dtype = args[1] if len(args) >= 2 else kwargs.get("dtype")
    device = args[2] if len(args) >= 3 else kwargs.get("device")
    return PhantomImpl(shape, dtype, device)


def _eye_factory(
    n: int, m: int = -1, k: int = 0, *args: Any, **kwargs: Any
) -> PhantomImpl:
    """Engine signature: ``eye(N, M=-1, k=0, dtype=F32, device=CPU, requires_grad=False)``."""
    del k
    rows = int(n)
    cols = int(m) if int(m) >= 0 else rows
    dtype = args[0] if len(args) >= 1 else kwargs.get("dtype")
    device = args[1] if len(args) >= 2 else kwargs.get("device")
    return PhantomImpl((rows, cols), dtype, device)


def _full_factory(shape: Any, value: Any, *args: Any, **kwargs: Any) -> PhantomImpl:
    """Engine signature: ``full(shape, value, dtype, device)``."""
    del value
    dtype = args[0] if len(args) >= 1 else kwargs.get("dtype")
    device = args[1] if len(args) >= 2 else kwargs.get("device")
    return PhantomImpl(shape, dtype, device)


def _randint_factory(
    shape: Any, low: Any, high: Any, *args: Any, **kwargs: Any
) -> PhantomImpl:
    """Engine signature: ``randint(shape, low, high, dtype=I64, device=CPU, generator=None)``."""
    del low, high
    dtype = args[0] if len(args) >= 1 else kwargs.get("dtype", _C_engine.I64)
    device = args[1] if len(args) >= 2 else kwargs.get("device")
    return PhantomImpl(shape, dtype, device)


def _uniform_normal_factory(
    shape: Any, p1: Any = 0.0, p2: Any = 1.0, *args: Any, **kwargs: Any
) -> PhantomImpl:
    """Engine signature: ``uniform/normal(shape, p1, p2, dtype=F32, device=CPU, generator=None)``."""
    del p1, p2
    dtype = args[0] if len(args) >= 1 else kwargs.get("dtype")
    device = args[1] if len(args) >= 2 else kwargs.get("device")
    return PhantomImpl(shape, dtype, device)


def _bernoulli_exponential_factory(
    shape: Any, param: Any = 1.0, *args: Any, **kwargs: Any
) -> PhantomImpl:
    """Engine signature: ``bernoulli/exponential(shape, p, dtype=F32, device=CPU, generator=None)``."""
    del param
    dtype = args[0] if len(args) >= 1 else kwargs.get("dtype")
    device = args[1] if len(args) >= 2 else kwargs.get("device")
    return PhantomImpl(shape, dtype, device)


def _arange_factory(
    start: float, end: float | None = None, step: float = 1, *args: Any, **kwargs: Any
) -> PhantomImpl:
    """Engine signature: ``arange(start, end, step, dtype, device)``.

    ``dtype`` and ``device`` are the two positions after ``step``, so
    they are ``args[0]`` and ``args[1]`` here. The length follows the
    engine's rule — ``ceil((end - start) / step)``, and none when the
    step points away from ``end`` — so the reported shape is the one a
    real run would give, fractional steps included.
    """
    if end is None:
        start, end = 0, start
    span = float(end) - float(start)
    length = math.ceil(span / step) if span * step > 0 else 0
    dtype = args[0] if len(args) >= 1 else kwargs.get("dtype")
    device = args[1] if len(args) >= 2 else kwargs.get("device")
    return PhantomImpl((length,), dtype, device)


def _linspace_factory(
    start: float, stop: float, num: int = 50, *args: Any, **kwargs: Any
) -> PhantomImpl:
    """Engine signature: ``linspace(start, stop, num=50, dtype=F32, device=CPU, requires_grad=False)``."""
    del start, stop
    dtype = args[0] if len(args) >= 1 else kwargs.get("dtype")
    device = args[1] if len(args) >= 2 else kwargs.get("device")
    return PhantomImpl((int(num),), dtype, device)


def _logspace_factory(
    start: float,
    stop: float,
    num: int = 50,
    base: float = 10.0,
    *args: Any,
    **kwargs: Any,
) -> PhantomImpl:
    """Engine signature: ``logspace(start, stop, num=50, base=10.0, dtype=F32, device=CPU, requires_grad=False)``."""
    del start, stop, base
    dtype = args[0] if len(args) >= 1 else kwargs.get("dtype")
    device = args[1] if len(args) >= 2 else kwargs.get("device")
    return PhantomImpl((int(num),), dtype, device)


def _astype_factory(impl: Any, dtype: Any, *args: Any, **kwargs: Any) -> Any:
    """``astype(impl, dtype)`` — cast a phantom's dtype metadata."""
    if isinstance(impl, PhantomImpl):
        return PhantomImpl(impl.shape, dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["astype"](impl, dtype, *args, **kwargs)


def _to_device_factory(impl: Any, device: Any, *args: Any, **kwargs: Any) -> Any:
    """``to_device(impl, device)`` — update a phantom's device metadata."""
    if isinstance(impl, PhantomImpl):
        return PhantomImpl(impl.shape, impl.dtype, device, impl.requires_grad)
    return _ORIGINAL["to_device"](impl, device, *args, **kwargs)


def _reshape_factory(impl: Any, shape: Any, *args: Any, **kwargs: Any) -> PhantomImpl:
    """``reshape(impl, shape)`` — used by ``init._fill_from_impl``.
    Pass-through for phantom input; defer to real engine otherwise."""
    if isinstance(impl, PhantomImpl):
        return PhantomImpl(
            _resolve_reshape(impl.shape, shape),
            impl.dtype,
            impl.device,
            impl.requires_grad,
        )
    return cast(PhantomImpl, _ORIGINAL["reshape"](impl, shape, *args, **kwargs))


def _view_factory(impl: Any, shape: Any, *args: Any, **kwargs: Any) -> Any:
    """``view(impl, shape)`` — resolves ``-1`` like ``reshape``."""
    if isinstance(impl, PhantomImpl):
        return PhantomImpl(
            _resolve_reshape(impl.shape, shape),
            impl.dtype,
            impl.device,
            impl.requires_grad,
        )
    return _ORIGINAL["view"](impl, shape, *args, **kwargs)


def _unsqueeze_factory(impl: Any, dim: int, *args: Any, **kwargs: Any) -> Any:
    """``unsqueeze(impl, dim)`` — insert a size-1 axis at ``dim``."""
    if isinstance(impl, PhantomImpl):
        ndim = len(impl.shape) + 1
        d = int(dim) % ndim
        new_shape = (*impl.shape[:d], 1, *impl.shape[d:])
        return PhantomImpl(new_shape, impl.dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["unsqueeze"](impl, dim, *args, **kwargs)


def _squeeze_factory(impl: Any, dim: int, *args: Any, **kwargs: Any) -> Any:
    """``squeeze(impl, dim)`` — remove size-1 axis ``dim``."""
    if isinstance(impl, PhantomImpl):
        if not impl.shape:
            return PhantomImpl((), impl.dtype, impl.device, impl.requires_grad)
        d = int(dim) % len(impl.shape)
        new_shape = (
            (*impl.shape[:d], *impl.shape[d + 1 :])
            if impl.shape[d] == 1
            else impl.shape
        )
        return PhantomImpl(new_shape, impl.dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["squeeze"](impl, dim, *args, **kwargs)


def _squeeze_all_factory(impl: Any, *args: Any, **kwargs: Any) -> Any:
    """``squeeze_all(impl)`` — drop all size-1 axes."""
    if isinstance(impl, PhantomImpl):
        new_shape = tuple(s for s in impl.shape if s != 1)
        return PhantomImpl(new_shape, impl.dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["squeeze_all"](impl, *args, **kwargs)


def _transpose_factory(impl: Any, *args: Any, **kwargs: Any) -> Any:
    """``transpose(impl)`` / ``T(impl)`` — reverse all axes."""
    if isinstance(impl, PhantomImpl):
        return PhantomImpl(
            impl.shape[::-1], impl.dtype, impl.device, impl.requires_grad
        )
    return _ORIGINAL["transpose"](impl, *args, **kwargs)


def _mT_factory(impl: Any, *args: Any, **kwargs: Any) -> Any:
    """``mT(impl)`` — swap the last two axes."""
    if isinstance(impl, PhantomImpl):
        s = impl.shape
        new_shape = (*s[:-2], s[-1], s[-2]) if len(s) >= 2 else s
        return PhantomImpl(new_shape, impl.dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["mT"](impl, *args, **kwargs)


def _swapaxes_factory(
    impl: Any, dim1: int, dim2: int, *args: Any, **kwargs: Any
) -> Any:
    """``swapaxes(impl, dim1, dim2)`` — swap two axes."""
    if isinstance(impl, PhantomImpl):
        s = list(impl.shape)
        if s:
            d1, d2 = int(dim1) % len(s), int(dim2) % len(s)
            s[d1], s[d2] = s[d2], s[d1]
        return PhantomImpl(tuple(s), impl.dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["swapaxes"](impl, dim1, dim2, *args, **kwargs)


def _permute_factory(impl: Any, dims: Any, *args: Any, **kwargs: Any) -> Any:
    """``permute(impl, dims)`` — reorder axes according to ``dims``."""
    if isinstance(impl, PhantomImpl):
        ndim = len(impl.shape)
        new_shape = tuple(impl.shape[int(d) % ndim] for d in dims) if ndim else ()
        return PhantomImpl(new_shape, impl.dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["permute"](impl, dims, *args, **kwargs)


def _flatten_factory(
    impl: Any, start_axis: int = 0, end_axis: int = -1, *args: Any, **kwargs: Any
) -> Any:
    """``flatten(impl, start_axis=0, end_axis=-1)``."""
    if isinstance(impl, PhantomImpl):
        s = impl.shape
        if not s:
            return PhantomImpl((1,), impl.dtype, impl.device, impl.requires_grad)
        ndim = len(s)
        sa = int(start_axis) % ndim
        ea = int(end_axis) % ndim
        flat = math.prod(s[sa : ea + 1])
        new_shape = (*s[:sa], flat, *s[ea + 1 :])
        return PhantomImpl(new_shape, impl.dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["flatten"](impl, start_axis, end_axis, *args, **kwargs)


def _ravel_factory(impl: Any, *args: Any, **kwargs: Any) -> Any:
    """``ravel(impl)`` — 1-D flattened view."""
    if isinstance(impl, PhantomImpl):
        return PhantomImpl((impl.numel(),), impl.dtype, impl.device, impl.requires_grad)
    return _ORIGINAL["ravel"](impl, *args, **kwargs)


def _matmul_factory(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
    """``matmul(a, b)`` / ``mm(a, b)`` / ``bmm(a, b)`` — contraction shape and promoted dtype."""
    if isinstance(a, PhantomImpl) or isinstance(b, PhantomImpl):
        sa, sb = tuple(a.shape), tuple(b.shape)
        if len(sa) == 1 and len(sb) == 1:
            out_shape: tuple[int, ...] = ()
        elif len(sa) == 1:
            out_shape = (*sb[:-2], sb[-1])
        elif len(sb) == 1:
            out_shape = sa[:-1]
        else:
            batch = _broadcast_shapes(sa[:-2], sb[:-2])
            out_shape = (*batch, sa[-2], sb[-1])
        dt = _promote_two(a, b)
        req_grad = bool(a.requires_grad or b.requires_grad)
        return PhantomImpl(out_shape, dt, a.device, req_grad)
    return _ORIGINAL["matmul"](a, b, *args, **kwargs)


def _dot_factory(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
    """``dot(a, b)`` — 1-D inner product -> 0-D scalar."""
    if isinstance(a, PhantomImpl) or isinstance(b, PhantomImpl):
        dt = _promote_two(a, b)
        return PhantomImpl((), dt, a.device, bool(a.requires_grad or b.requires_grad))
    return _ORIGINAL["dot"](a, b, *args, **kwargs)


def _inner_factory(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
    """``inner(a, b)`` — contracts the last axis of ``a`` and ``b``."""
    if isinstance(a, PhantomImpl) or isinstance(b, PhantomImpl):
        sa, sb = tuple(a.shape), tuple(b.shape)
        out_shape = (*sa[:-1], *sb[:-1]) if (sa and sb) else (*sa, *sb)
        dt = _promote_two(a, b)
        return PhantomImpl(
            out_shape, dt, a.device, bool(a.requires_grad or b.requires_grad)
        )
    return _ORIGINAL["inner"](a, b, *args, **kwargs)


def _outer_factory(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
    """``outer(a, b)`` — 1-D × 1-D -> 2-D ``(a.numel(), b.numel())``."""
    if isinstance(a, PhantomImpl) or isinstance(b, PhantomImpl):
        dt = _promote_two(a, b)
        return PhantomImpl(
            (int(a.numel()), int(b.numel())),
            dt,
            a.device,
            bool(a.requires_grad or b.requires_grad),
        )
    return _ORIGINAL["outer"](a, b, *args, **kwargs)


def _kron_factory(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
    """``kron(a, b)`` — Kronecker product shape."""
    if isinstance(a, PhantomImpl) or isinstance(b, PhantomImpl):
        sa, sb = tuple(a.shape), tuple(b.shape)
        rank = max(len(sa), len(sb))
        pa = (1,) * (rank - len(sa)) + sa
        pb = (1,) * (rank - len(sb)) + sb
        out_shape = tuple(x * y for x, y in zip(pa, pb, strict=True))
        dt = _promote_two(a, b)
        return PhantomImpl(
            out_shape, dt, a.device, bool(a.requires_grad or b.requires_grad)
        )
    return _ORIGINAL["kron"](a, b, *args, **kwargs)


def _where_factory(cond: Any, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
    """``where(cond, x, y)`` — 3-way broadcast shape and branch dtype."""
    if (
        isinstance(cond, PhantomImpl)
        or isinstance(x, PhantomImpl)
        or isinstance(y, PhantomImpl)
    ):
        shape = _broadcast_shapes(
            _broadcast_shapes(tuple(cond.shape), tuple(x.shape)), tuple(y.shape)
        )
        dt = _promote_two(x, y)
        req_grad = bool(x.requires_grad or y.requires_grad)
        return PhantomImpl(shape, dt, x.device, req_grad)
    return _ORIGINAL["where"](cond, x, y, *args, **kwargs)


def _cat_factory(tensors: Any, dim: int = 0, *args: Any, **kwargs: Any) -> Any:
    """``concatenate(tensors, dim=0)`` / ``concat(tensors, dim)``."""
    if isinstance(tensors, (list, tuple)) and any(
        isinstance(t, PhantomImpl) for t in tensors
    ):
        rep = next(t for t in tensors if isinstance(t, PhantomImpl))
        base = list(rep.shape)
        if base:
            d = int(dim) % len(base)
            base[d] = sum(int(t.shape[d]) for t in tensors)
        return PhantomImpl(tuple(base), rep.dtype, rep.device, rep.requires_grad)
    return _ORIGINAL["concatenate"](tensors, dim, *args, **kwargs)


def _stack_factory(tensors: Any, dim: int = 0, *args: Any, **kwargs: Any) -> Any:
    """``stack(tensors, dim=0)``."""
    if isinstance(tensors, (list, tuple)) and any(
        isinstance(t, PhantomImpl) for t in tensors
    ):
        rep = next(t for t in tensors if isinstance(t, PhantomImpl))
        ndim = len(rep.shape) + 1
        d = int(dim) % ndim
        out_shape = (*rep.shape[:d], len(tensors), *rep.shape[d:])
        return PhantomImpl(out_shape, rep.dtype, rep.device, rep.requires_grad)
    return _ORIGINAL["stack"](tensors, dim, *args, **kwargs)


def _hstack_factory(tensors: Any, *args: Any, **kwargs: Any) -> Any:
    """``hstack(tensors)`` — concatenate along axis 0 for 1-D, axis 1 for >=2-D."""
    if isinstance(tensors, (list, tuple)) and any(
        isinstance(t, PhantomImpl) for t in tensors
    ):
        rep = next(t for t in tensors if isinstance(t, PhantomImpl))
        dim = 0 if len(rep.shape) <= 1 else 1
        return _cat_factory(tensors, dim)
    return _ORIGINAL["hstack"](tensors, *args, **kwargs)


def _vstack_factory(tensors: Any, *args: Any, **kwargs: Any) -> Any:
    """``vstack(tensors)`` — stack 1-D inputs into ``(N, L)``, concatenate >=2-D along axis 0."""
    if isinstance(tensors, (list, tuple)) and any(
        isinstance(t, PhantomImpl) for t in tensors
    ):
        rep = next(t for t in tensors if isinstance(t, PhantomImpl))
        if len(rep.shape) == 1:
            return _stack_factory(tensors, 0)
        return _cat_factory(tensors, 0)
    return _ORIGINAL["vstack"](tensors, *args, **kwargs)


def _complex_factory(real: Any, imag: Any, *args: Any, **kwargs: Any) -> Any:
    """``complex(real, imag)`` — broadcast shape and complex64/complex128 dtype."""
    if isinstance(real, PhantomImpl) or isinstance(imag, PhantomImpl):
        shape = _broadcast_shapes(tuple(real.shape), tuple(imag.shape))
        part_dt = _promote_two(real, imag)
        dt = _C_engine.C128 if part_dt == _C_engine.F64 else _C_engine.C64
        req_grad = bool(real.requires_grad or imag.requires_grad)
        return PhantomImpl(shape, dt, real.device, req_grad)
    return _ORIGINAL["complex"](real, imag, *args, **kwargs)


def _real_imag_factory(a: Any, *args: Any, **kwargs: Any) -> Any:
    """``real(a)`` / ``imag(a)`` — extract real floating dtype from complex."""
    if isinstance(a, PhantomImpl):
        if a.dtype == _C_engine.C128:
            dt = _C_engine.F64
        elif a.dtype == _C_engine.C64:
            dt = _C_engine.F32
        else:
            dt = a.dtype
        return PhantomImpl(a.shape, dt, a.device, a.requires_grad)
    return _ORIGINAL["real"](a, *args, **kwargs)


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
        combined = tuple(
            t.shape[0] if isinstance(t, PhantomImpl) and t.shape else 1 for t in tensors
        )
        rep = next(t for t in tensors if isinstance(t, PhantomImpl))
        return [PhantomImpl(combined, rep.dtype, rep.device, False) for _ in tensors]
    return cast(list[Any], _ORIGINAL["meshgrid"](tensors, *args, **kwargs))


_OPMATH_BINARY_OPS = frozenset({"mul", "mul_", "div", "div_", "floordiv"})


def _binary_wrap(
    orig_name: str,
    orig: Any,
    *,
    out_bool: bool = False,
    float_integral: bool = False,
    inplace: bool = False,
) -> Any:
    """Wrapper for binary elementwise engine ops ``f(a, b)``."""
    is_opmath = orig_name in _OPMATH_BINARY_OPS

    def w(a: Any, b: Any, *rest: Any, **kwargs: Any) -> Any:
        if isinstance(a, PhantomImpl) or isinstance(b, PhantomImpl):
            shape = _broadcast_shapes(tuple(a.shape), tuple(b.shape))
            if out_bool:
                dt = _C_engine.Bool
            else:
                dt = _promote_two(a, b, true_div=float_integral, opmath=is_opmath)
            if inplace and isinstance(a, PhantomImpl):
                a.shape = shape
                a.dtype = dt
                a.ndim = len(shape)
                return a
            req_grad = bool(a.requires_grad or b.requires_grad)
            return PhantomImpl(shape, dt, a.device, req_grad)
        return orig(a, b, *rest, **kwargs)

    return w


def _unary_float_wrap(orig: Any) -> Any:
    """Wrapper for unary math ops that promote integral/bool inputs to default float."""
    from lucid import _promotion

    def w(a: Any, *rest: Any, **kwargs: Any) -> Any:
        if isinstance(a, PhantomImpl):
            dt = (
                _promotion.default_float(a.device)
                if _promotion.is_integral(a.dtype)
                else a.dtype
            )
            return PhantomImpl(a.shape, dt, a.device, a.requires_grad)
        return orig(a, *rest, **kwargs)

    return w


def _unary_bool_wrap(orig: Any) -> Any:
    """Wrapper for unary predicate ops that always return ``Bool``."""

    def w(a: Any, *rest: Any, **kwargs: Any) -> Any:
        if isinstance(a, PhantomImpl):
            return PhantomImpl(a.shape, _C_engine.Bool, a.device, False)
        return orig(a, *rest, **kwargs)

    return w


def _reduce_dtype(in_dtype: Any, device: Any, kind: str) -> Any:
    """Compute output dtype for a reduction op of ``kind``."""
    from lucid import _promotion

    if kind == "int64_integral":
        return _C_engine.I64 if _promotion.is_integral(in_dtype) else in_dtype
    if kind == "float_integral":
        return (
            _promotion.default_float(device)
            if _promotion.is_integral(in_dtype)
            else in_dtype
        )
    if kind == "int64":
        return _C_engine.I64
    if kind == "bool":
        return _C_engine.Bool
    return in_dtype


def _reduce_wrap(orig: Any, *, kind: str) -> Any:
    """Wrapper for reduction ops ``f(a, dim=[], keepdims=False)``."""

    def w(
        a: Any, dim: Any = None, keepdims: bool = False, *rest: Any, **kwargs: Any
    ) -> Any:
        if isinstance(a, PhantomImpl):
            shape = _reduced_shape(a.shape, dim, keepdims)
            dt = _reduce_dtype(a.dtype, a.device, kind)
            return PhantomImpl(shape, dt, a.device, a.requires_grad)
        if dim is None:
            return orig(a, *rest, **kwargs)
        return orig(a, dim, keepdims, *rest, **kwargs)

    return w


def _cum_int64_wrap(orig: Any) -> Any:
    """Wrapper for ``cumsum`` / ``cumprod`` which widen integral/bool inputs to ``int64``."""
    from lucid import _promotion

    def w(a: Any, *rest: Any, **kwargs: Any) -> Any:
        if isinstance(a, PhantomImpl):
            dt = _C_engine.I64 if _promotion.is_integral(a.dtype) else a.dtype
            return PhantomImpl(a.shape, dt, a.device, a.requires_grad)
        return orig(a, *rest, **kwargs)

    return w


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


def _shape_preserving_wrap(orig: Any) -> Any:
    """Default phantom-aware wrapper: if any arg (including elements of
    a top-level list/tuple) is a phantom, return a phantom with that
    arg's shape.  Falls through to the real engine when nothing
    phantom is present so non-shadow paths in the same process aren't
    affected."""

    def w(*args: Any, **kwargs: Any) -> Any:
        ph = _find_phantom(args)
        if ph is not None:
            return PhantomImpl(ph.shape, ph.dtype, ph.device, ph.requires_grad)
        return orig(*args, **kwargs)

    return w


def _explicit_shape_wrap(orig: Any) -> Any:
    """For ops whose output shape = **second positional arg** (the
    shape parameter): ``broadcast_to(t, shape)``, ``expand(t, shape)``."""

    def w(impl: Any, shape: Any, *rest: Any, **kwargs: Any) -> Any:
        if isinstance(impl, PhantomImpl):
            return PhantomImpl(shape, impl.dtype, impl.device, impl.requires_grad)
        return orig(impl, shape, *rest, **kwargs)

    return w


# Engine ops whose output shape is the *explicit* shape argument (not
# derived from input).
_EXPLICIT_SHAPE_OPS = frozenset(
    {
        "broadcast_to",
        "expand",
    }
)

_BINARY_ARITH_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "floordiv",
        "pow",
        "maximum",
        "minimum",
        "fmod",
        "remainder",
        "hypot",
        "logaddexp",
        "nextafter",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
    }
)

_BINARY_INPLACE_OPS = frozenset(
    {
        "add_",
        "sub_",
        "mul_",
        "div_",
        "pow_",
        "maximum_",
        "minimum_",
    }
)

_BINARY_FLOAT_OPS = frozenset({"div", "atan2"})

_BINARY_BOOL_OPS = frozenset(
    {
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "logical_and",
        "logical_or",
        "logical_xor",
        "isclose",
    }
)

_UNARY_FLOAT_OPS = frozenset(
    {
        "exp",
        "exp2",
        "log",
        "log2",
        "log10",
        "log1p",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "sigmoid",
        "silu",
        "gelu",
        "gelu_exact",
        "mish",
        "selu",
        "softplus",
        "hard_sigmoid",
        "hard_swish",
        "erf",
        "erfinv",
    }
)

_UNARY_BOOL_OPS = frozenset({"isnan", "isinf", "isfinite", "logical_not"})

_REDUCE_KIND_BY_OP: dict[str, str] = {
    "sum": "int64_integral",
    "prod": "int64_integral",
    "mean": "float_integral",
    "var": "float_integral",
    "std": "float_integral",
    "logsumexp": "float_integral",
    "max": "same",
    "min": "same",
    "argmax": "int64",
    "argmin": "int64",
    "any": "bool",
    "all": "bool",
}

_CUM_INT64_OPS = frozenset({"cumsum", "cumprod"})

_BINARY_OP_KWARGS: dict[str, dict[str, bool]] = {
    **{op: {} for op in _BINARY_ARITH_OPS},
    **{op: {"inplace": True} for op in _BINARY_INPLACE_OPS},
    **{op: {"float_integral": True} for op in _BINARY_FLOAT_OPS},
    **{op: {"out_bool": True} for op in _BINARY_BOOL_OPS},
}

_UNARY_WRAPPER_BY_OP: dict[str, Any] = {
    **{op: _explicit_shape_wrap for op in _EXPLICIT_SHAPE_OPS},
    **{op: _unary_float_wrap for op in _UNARY_FLOAT_OPS},
    **{op: _unary_bool_wrap for op in _UNARY_BOOL_OPS},
    **{op: _cum_int64_wrap for op in _CUM_INT64_OPS},
}

# Engine functions whose semantics need a custom wrapper (not the
# generic shape-preserving default).
_SPECIFIC_WRAPPERS: dict[str, Any] = {
    "zeros": _shape_only_factory,
    "ones": _shape_only_factory,
    "empty": _shape_only_factory,
    "eye": _eye_factory,
    "randn": _shape_only_factory,
    "rand": _shape_only_factory,
    "randint": _randint_factory,
    "uniform": _uniform_normal_factory,
    "normal": _uniform_normal_factory,
    "bernoulli": _bernoulli_exponential_factory,
    "exponential": _bernoulli_exponential_factory,
    "full": _full_factory,
    "arange": _arange_factory,
    "linspace": _linspace_factory,
    "logspace": _logspace_factory,
    "astype": _astype_factory,
    "to_device": _to_device_factory,
    "reshape": _reshape_factory,
    "view": _view_factory,
    "unsqueeze": _unsqueeze_factory,
    "expand_dims": _unsqueeze_factory,
    "squeeze": _squeeze_factory,
    "squeeze_all": _squeeze_all_factory,
    "transpose": _transpose_factory,
    "T": _transpose_factory,
    "mT": _mT_factory,
    "swapaxes": _swapaxes_factory,
    "permute": _permute_factory,
    "flatten": _flatten_factory,
    "ravel": _ravel_factory,
    "matmul": _matmul_factory,
    "mm": _matmul_factory,
    "bmm": _matmul_factory,
    "dot": _dot_factory,
    "inner": _inner_factory,
    "outer": _outer_factory,
    "kron": _kron_factory,
    "where": _where_factory,
    "concatenate": _cat_factory,
    "concat": _cat_factory,
    "stack": _stack_factory,
    "hstack": _hstack_factory,
    "vstack": _vstack_factory,
    "complex": _complex_factory,
    "real": _real_imag_factory,
    "imag": _real_imag_factory,
    "meshgrid": _meshgrid_factory,
}

# Engine name -> original callable, filled while the stand-ins are installed.
# The factories above fall back to it for an argument that is not a phantom.
_ORIGINAL: dict[str, Any] = {}


def _make_engine_wrapper(name: str, attr: Any) -> Any:
    """Select the shadow wrapper for ``_C_engine.<name>``."""
    if name in _SPECIFIC_WRAPPERS:
        return _SPECIFIC_WRAPPERS[name]
    if name in _BINARY_OP_KWARGS:
        return _binary_wrap(name, attr, **_BINARY_OP_KWARGS[name])
    if name in _REDUCE_KIND_BY_OP:
        return _reduce_wrap(attr, kind=_REDUCE_KIND_BY_OP[name])
    builder = _UNARY_WRAPPER_BY_OP.get(name, _shape_preserving_wrap)
    return builder(attr)


# ---------------------------------------------------------------------------
# Installing and removing the stand-ins
# ---------------------------------------------------------------------------
#
# The stand-ins replace module attributes, so they are process-wide; the
# first block to open anywhere installs them and the last to close removes
# them.  A process with no open block runs the untouched engine.

_INSTALL_LOCK = threading.Lock()
_open_blocks = 0
# id(stand-in) -> (stand-in, original).  Holding the stand-in keeps its id
# from being reused, so the identity scan on removal cannot misfire.
_INSTALLED: dict[int, tuple[object, Any]] = {}


def _gated[**P, R](orig: Callable[P, R], shadow: Callable[P, R]) -> Callable[P, R]:
    """Return a stand-in for ``orig`` that runs ``shadow`` only inside a block.

    Every stand-in this module installs is made here, so a thread that is
    not inside a block always reaches ``orig``.
    """
    active = _ACTIVE.get

    def stand_in(*args: P.args, **kwargs: P.kwargs) -> R:
        if active():
            return shadow(*args, **kwargs)
        return orig(*args, **kwargs)

    _INSTALLED[id(stand_in)] = (stand_in, orig)
    return stand_in


def _module_namespaces() -> list[dict[str, Any]]:
    """Return the namespace of every loaded module."""
    # Typed ``object``: anything can be put in ``sys.modules``, not only a
    # module.
    loaded: list[object] = list(sys.modules.values())
    # ``object.__getattribute__`` reads ``__dict__`` without running a lazy
    # module's attribute hook, which would import it.
    return [
        object.__getattribute__(mod, "__dict__")
        for mod in loaded
        if isinstance(mod, types.ModuleType)
    ]


def _swap_closure_cells(fn: object, replacement: dict[int, Any]) -> None:
    """Rebind each closure cell of ``fn`` whose value has a replacement."""
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:  # an empty cell: its variable is not bound yet
            continue
        new = replacement.get(id(value))
        if new is not None:
            cell.cell_contents = new


def _swap_op_registry(replacement: dict[int, Any]) -> None:
    """Rebind ``OpEntry.engine_fn`` and the adapter closure cells by identity."""
    from lucid._ops import _adapters as _adapters_mod
    from lucid._ops._registry import _REGISTRY as _OP_REGISTRY

    for entry in _OP_REGISTRY:
        new = replacement.get(id(entry.engine_fn))
        if new is not None:
            entry.engine_fn = new
            continue
        _swap_closure_cells(entry.engine_fn, replacement)
    _swap_closure_cells(_adapters_mod._complex_parts, replacement)


def _phantom_operand(x: object) -> PhantomImpl | None:
    """Return the phantom ``x`` is or wraps, else ``None``."""
    impl = getattr(x, "_impl", None)
    if isinstance(impl, PhantomImpl):
        return impl
    if isinstance(x, PhantomImpl):
        return x
    return None


def _install_unwrap_stand_ins() -> None:
    """Pass phantoms through every module's ``_unwrap`` / ``_unwrap_or_scalar`` / ``try_unwrap``."""
    from lucid import _dispatch as _disp
    from lucid import _promotion as _promo

    orig_unwrap = _disp._unwrap
    orig_uos = _disp._unwrap_or_scalar
    orig_try = _promo.try_unwrap

    def unwrap(t: Any) -> Any:
        phantom = _phantom_operand(t)
        return orig_unwrap(t) if phantom is None else phantom

    def unwrap_or_scalar(x: Any, ref_impl: Any = None) -> Any:
        phantom = _phantom_operand(x)
        return orig_uos(x, ref_impl) if phantom is None else phantom

    def try_unwrap(x: object) -> object:
        phantom = _phantom_operand(x)
        return orig_try(x) if phantom is None else phantom

    stand_in_try = _gated(orig_try, try_unwrap)
    stand_ins = {
        "_unwrap": (orig_unwrap, _gated(orig_unwrap, unwrap)),
        "_unwrap_or_scalar": (orig_uos, _gated(orig_uos, unwrap_or_scalar)),
        "try_unwrap": (orig_try, stand_in_try),
        "_try_unwrap": (orig_try, stand_in_try),
    }
    for ns in (vars(_promo), *_module_namespaces()):
        for name, (orig, stand_in) in stand_ins.items():
            if ns.get(name) is orig:
                ns[name] = stand_in


def _install() -> None:
    """Install a gated stand-in for every public engine callable and its users."""
    _install_unwrap_stand_ins()
    stand_in_of: dict[int, Any] = {}
    for name in dir(_C_engine):
        if name.startswith("_"):
            continue
        attr = getattr(_C_engine, name)
        if not callable(attr) or isinstance(attr, type):
            continue
        _ORIGINAL[name] = attr
        stand_in = _gated(attr, _make_engine_wrapper(name, attr))
        stand_in_of[id(attr)] = stand_in
        setattr(_C_engine, name, stand_in)
    _swap_op_registry(stand_in_of)


def _uninstall() -> None:
    """Put the original back wherever a stand-in is bound.

    The search is by identity over every loaded module, so a module first
    imported inside a block — which bound a stand-in — is restored too.
    """
    from lucid import _dispatch as _disp
    from lucid import _promotion as _promo

    original_of = {key: orig for key, (_stand_in, orig) in _INSTALLED.items()}
    ids = original_of.keys()
    # The source modules go first: an import racing this then binds an
    # original.
    for ns in (vars(_C_engine), vars(_disp), vars(_promo), *_module_namespaces()):
        if ids.isdisjoint(map(id, list(ns.values()))):
            continue
        for name, value in list(ns.items()):
            orig = original_of.get(id(value))
            if orig is not None:
                ns[name] = orig
    _swap_op_registry(original_of)
    _INSTALLED.clear()
    _ORIGINAL.clear()


def _open_block() -> None:
    """Count a block opening, installing the stand-ins for the first one."""
    global _open_blocks
    with _INSTALL_LOCK:
        if _open_blocks == 0:
            installed = False
            try:
                _install()
                installed = True
            finally:
                if not installed:
                    _uninstall()
        _open_blocks += 1


def _close_block() -> None:
    """Count a block closing, removing the stand-ins after the last one."""
    global _open_blocks
    with _INSTALL_LOCK:
        _open_blocks -= 1
        if _open_blocks == 0:
            _uninstall()


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------


def is_active() -> bool:
    """Return whether the calling thread is inside a :func:`shadow_alloc` block.

    Used by Lucid's introspection paths (param-count helpers, layer
    summary generators, the docs build's ``shadow_alloc=True`` factory
    invocations) to decide whether tensor storage is real or
    metadata-only.

    Returns
    -------
    bool
        ``True`` inside a :func:`shadow_alloc` block opened by the calling
        thread, ``False`` otherwise — even while another thread has one
        open.
    """
    return _ACTIVE.get()


def is_shadow(t: Tensor) -> bool:
    """Return whether ``t`` is a shadow tensor, carrying metadata but no storage.

    Parameters
    ----------
    t : Tensor
        The tensor to inspect.

    Returns
    -------
    bool
        ``True`` when ``t`` was made inside a :func:`shadow_alloc` block,
        ``False`` for a tensor with real storage.
    """
    return isinstance(t._impl, PhantomImpl)


@contextlib.contextmanager
def shadow_alloc() -> Iterator[None]:
    r"""Construct models without allocating any tensor storage.

    Use as a context manager around factory invocations whose only
    purpose is to inspect structure (e.g. layer-summary generation
    for the docs site).  Inside the block the engine's tensor-creation
    entry points return phantom impls; no real memory is reserved
    even for hundreds-of-billions-parameter models.

    The block applies to the calling thread only.  Other threads,
    including ones started inside it, keep allocating and computing
    real values, and blocks open in two threads at once each last
    until their own exit.  A block nested inside another in the same
    thread does nothing of its own.

    Examples
    --------
    >>> from lucid.nn._shadow import shadow_alloc
    >>> from lucid.models.vision.resnet import resnet_152
    >>> with shadow_alloc():
    ...     model = resnet_152()  # constructs without GB of RAM
    >>> # Only structural metadata is meaningful afterwards:
    >>> import math
    >>> sum(math.prod(p.shape) for p in model.parameters())  # counted from shapes
    58143808
    """
    if _ACTIVE.get():
        yield
        return
    _open_block()
    try:
        token = _ACTIVE.set(True)
        try:
            yield
        finally:
            _ACTIVE.reset(token)
    finally:
        _close_block()
