"""Adapter functions for ops whose Python signature differs from the engine's.

Every function here normalises a flexible Python calling convention (e.g.
``dim=None | int | list``, ``keepdim=...``, variadic shape, list-as-positional)
into the strict positional signature that the underlying engine kernel
expects.  ``_registry.py`` references these adapters as ``engine_fn`` on the
relevant ``OpEntry``; ``_make_method`` and ``_make_free_fn`` then forward
both ``*args`` and ``**kwargs`` through, so the adapter sees exactly what
the user wrote at the call site.

Adapters fall into a few buckets:

* ``_*_adapter``           — translate Python signatures over an existing
                              engine kernel (sum/mean/squeeze/etc.).
* composite shim adapters  — pre/post-process around a ``composite/`` op
                              (scatter, take, index_select, ...).
* sub-module forwarders    — route into ``lucid.linalg`` / ``lucid.einops``
                              for top-level aliases (cross, norm, einsum).

The reduction adapters take their axes from :mod:`lucid._ops._reduce`, which
also holds ``var`` / ``std`` and the numerics they share with the norms.
"""

import inspect
import math
import operator
from typing import Callable, Sequence, TYPE_CHECKING, cast

from lucid._C import engine as _C_engine
from lucid._deprecation import warn_deprecated
from lucid import _promotion
from lucid._dispatch import (
    _REAL_FLOATS,
    _binary_operands,
    _promote_impls,
    _refuse_bool_subtraction,
    _scalar_power,
    _unwrap,
    _unwrap_or_scalar,
)
from lucid._dtype import _ENGINE_TO_DTYPE, to_engine_dtype
from lucid._ops._reduce import _reduction_axes
from lucid._types import Scalar, TensorOrScalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# Convenience alias for the engine's TensorImpl class — it's the lingua
# franca of every adapter signature.  Adapters consume the engine's raw
# storage type, while user-facing args are typed as ``Tensor``.
_Impl = _C_engine.TensorImpl


def _own_impl(out: _Impl, a_impl: _Impl) -> _Impl:
    """``out``, or a same-shape view of ``a_impl`` when an op handed it back.

    A no-op view (``narrow`` over a whole dimension, ``squeeze`` of a dim
    that is not size 1) returned its input's own ``TensorImpl``, so two
    Python tensors shared one: flipping ``requires_grad`` on the result
    flipped it on the input, and a gradient accumulated into either landed
    on both.  A view is a tensor of its own — the reference framework's
    result is a distinct view too — that still reads and writes the input's
    storage and, when the input requires grad, carries it through autograd.
    """
    if out is a_impl:
        return _C_engine.view(a_impl, list(a_impl.shape))
    return out


def _keep_engine_signature(
    adapter: Callable[..., object], engine_fn: Callable[..., object]
) -> None:
    """Let ``adapter`` stand in for ``engine_fn`` wherever a signature is read.

    The pybind11 docstring carries the canonical signature line, and
    ``__wrapped__`` points ``_signature_for_entry`` back at it, so
    ``inspect.signature``, ``help()`` and ``gen_pyi.py`` see the op's own
    Tensor-typed parameters rather than the adapter's ``_Impl``-typed ones:
    wrapping an op leaves its public signature and its stub as they were.
    """
    adapter.__doc__ = getattr(engine_fn, "__doc__", None)
    adapter.__name__ = getattr(engine_fn, "__name__", adapter.__name__)
    adapter.__wrapped__ = engine_fn  # type: ignore[attr-defined]


#: The engine ops that compute a half tensor against a 0-d float32 operand
#: in float32 (op-math); the reference does so for these three only.
_OPMATH_OPS = (
    _C_engine.mul,
    _C_engine.mul_,
    _C_engine.div,
    _C_engine.div_,
    _C_engine.floordiv,
)


#: ``(input, other)`` without annotations: either operand may be a tensor or
#: a Python number, which the adapter's ``_Impl`` annotations do not say.
_BINARY_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter("input", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("other", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
)


def _make_arith_adapter(
    engine_fn: Callable[[_Impl, _Impl], _Impl],
    *,
    floating: bool = False,
    inplace: bool = False,
) -> Callable[[_Impl, _Impl], _Impl]:
    """Wrap a binary engine function with scalars and dtype promotion.

    Both operands go through :func:`lucid._dispatch._binary_operands`, the
    path the operators take too, so ``a.add(2)``, ``lucid.add(a, 2)`` and
    ``a + 2`` cannot drift apart: a Python number becomes a 0-d constant, and
    the pair meets at the dtype :mod:`lucid._promotion` gives it.  Two tensors
    of one dtype pass straight through.

    Used by the arithmetic ops (add/sub/mul/div/pow/maximum/minimum,
    floor_divide) and by the elementwise binaries the reference framework
    also accepts a scalar for: the comparisons, fmod/remainder, and the
    bitwise and shift ops.  Those return their own dtype — Bool for a
    comparison — but they still want both operands at a common one first.

    matmul/dot/inner/outer deliberately bypass it: a scalar operand is
    meaningless there.  ``outer`` promotes its two tensors through
    ``_outer_adapter`` instead; the others have dtype constraints of their
    own that the engine enforces.

    ``floating`` adds true division's rule on top: a common dtype that is
    integral or bool becomes the default float dtype, as it does for ``/``.
    mul, div and floor division — in place or not — hand a half tensor's
    scalar partner over unrounded, for the engine's op-math.

    ``inplace`` marks the trailing-underscore form, which writes the result
    back into ``input``.  Beside ``floating`` it refuses an integral one: the
    quotient is floating, and an integer tensor cannot hold it in place.
    ``div_`` used to floor-divide in silence where the reference refuses —
    and where ``/=`` already refused.
    """
    name = getattr(engine_fn, "__name__", "_arith_adapter")

    subtracts = engine_fn in (_C_engine.sub, _C_engine.sub_)
    powers = engine_fn is _C_engine.pow
    opmath = engine_fn in _OPMATH_OPS

    def _adapter(input: _Impl, other: _Impl) -> _Impl:
        if powers and isinstance(input, _C_engine.TensorImpl):
            scalar = _scalar_power(input, other)
            if scalar is not None:
                return scalar
        if inplace and floating:
            target = _unwrap_or_scalar(input)
            if _promotion.is_integral(target.dtype):
                dtype = _ENGINE_TO_DTYPE.get(target.dtype, target.dtype)
                raise RuntimeError(
                    f"{name}: true division gives a floating result, which cannot "
                    f"be written in place into a {dtype} tensor — use "
                    f"{name.removesuffix('_')} instead"
                )
        a, b = _binary_operands(input, other, true_div=floating, opmath=opmath)
        if subtracts:
            _refuse_bool_subtraction(a, b)
        return engine_fn(a, b)

    # The adapter's own ``(input, other)`` is the op's signature — not the
    # engine's names, which for ``atan2(y, x)`` came out as ``(other, input)``
    # and advertised the operands in the wrong order.
    _adapter.__doc__ = getattr(engine_fn, "__doc__", None)
    _adapter.__name__ = name
    _adapter.__dict__["__signature__"] = _BINARY_SIGNATURE
    return _adapter


_complex_parts = _make_arith_adapter(_C_engine.complex)


def _complex_adapter(real: _Impl, imag: _Impl) -> _Impl:
    """complex(real, imag) — the reference's names for the two parts."""
    return _complex_parts(real, imag)


def _make_join_adapter(engine_fn: Callable[..., _Impl]) -> Callable[..., _Impl]:
    """Wrap a joining engine op (cat, stack, ...) so it takes mixed dtypes.

    The engine joins tensors of a single dtype only, so ``cat`` of an
    ``int64`` and a ``float32`` raised ``DtypeMismatch``, where the
    reference joins them at their common dtype.  The inputs are brought
    there first; inputs that already agree pass straight through.
    """

    name = engine_fn.__name__

    def _adapter(tensors: Sequence[_Impl], *args: object, **kwargs: object) -> _Impl:
        # Looked up by name on each call rather than captured: shadow
        # allocation swaps the engine's attributes for phantom-aware stand-ins
        # and rebinds only the registry entries that hold the bare engine op,
        # so an adapter holding the original would hand it phantoms it rejects.
        join = cast(Callable[..., _Impl], getattr(_C_engine, name))
        return join(_promote_impls(tensors), *args, **kwargs)

    _keep_engine_signature(_adapter, engine_fn)
    return _adapter


# ── Engine-arg-order adapters ────────────────────────────────────────────────


def _detach_adapter(x_impl: _Impl) -> _Impl:
    """detach(x): the same storage, cut from the graph (see Tensor.detach)."""
    return x_impl.data_alias()


def _scatter_add_adapter(
    x_impl: _Impl,
    dim: int,
    index: Tensor,
    src: Tensor,
) -> _Impl:
    """scatter_add(x, dim, index, src) — Python order → engine order.

    The index goes over at full width: the engine checks an ``int64`` index
    against the axis as it is, and narrowing it to ``int32`` first wrapped
    ``2**40`` round to ``0`` and added into the wrong slot.
    """
    return _C_engine.scatter_add(x_impl, _unwrap(index), _unwrap(src), dim)


#: ``gather`` took ``(input, indices, dim=-1)`` until 3.16.  The reference
#: order ``(input, dim, index)`` replaced it; the old one still answers, and
#: warns, until the release named here.
_GATHER_OLD_ORDER = "the argument order gather(input, indices, dim)"
_GATHER_SINCE = "3.16.0"
_GATHER_REMOVAL = "3.18.0"


def _as_dim(value: object) -> int:
    """``value`` as a ``gather`` dim: an int, or a 0-d integer tensor."""
    impl = getattr(value, "_impl", None)
    if isinstance(impl, _Impl):
        if not impl.shape and _promotion.is_integer(impl.dtype):
            return int(cast(int, impl.item()))
        raise TypeError(
            "gather(): dim must be an int or a 0-d integer tensor, got a tensor "
            f"of shape {tuple(impl.shape)} and dtype {_ENGINE_TO_DTYPE[impl.dtype]}"
        )
    try:
        return operator.index(value)  # type: ignore[arg-type]  # refused below
    except TypeError:
        raise TypeError(
            f"gather(): dim must be an int, got {type(value).__name__}"
        ) from None


def _gather_operands(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[int, Tensor]:
    """``(dim, index)`` from whichever spelling of ``gather`` the caller used.

    The reference order is ``gather(input, dim, index)``.  Lucid's was
    ``gather(input, indices, dim=-1)`` and released code still calls it so,
    which no single signature can bind alongside the reference one — the
    argument after ``input`` decides instead.  An integer there is a
    ``dim``: the reference order.  So is a 0-d tensor followed by a tensor
    index, which the reference reads as a dim too.  Any other tensor there,
    or the old keyword ``indices=``, is the old order, which still answers
    and warns.
    """
    extra = sorted(set(kwargs) - {"dim", "index", "indices"})
    if extra:
        raise TypeError(f"gather() got an unexpected keyword argument {extra[0]!r}")
    if len(args) > 2:
        raise TypeError(
            f"gather() takes 3 positional arguments but {len(args) + 1} were given"
        )
    second = getattr(args[0], "_impl", None) if args else None
    index_follows = "index" in kwargs or (len(args) > 1 and hasattr(args[1], "_impl"))
    zero_d_dim = isinstance(second, _Impl) and not second.shape and index_follows
    old_order = "indices" in kwargs or (second is not None and not zero_d_dim)
    if old_order and "index" in kwargs:
        raise TypeError("gather() got multiple values for argument 'index'")

    names = ("indices", "dim") if old_order else ("dim", "index")
    bound: dict[str, object] = {"dim": -1} if old_order else {}
    for name, value in zip(names, args):
        if name in kwargs:
            raise TypeError(f"gather() got multiple values for argument {name!r}")
        bound[name] = value
    bound.update(kwargs)
    missing = [name for name in names if name not in bound]
    if missing:
        raise TypeError(f"gather() missing required argument {missing[0]!r}")

    index = bound["indices" if old_order else "index"]
    if not hasattr(index, "_impl"):
        raise TypeError(f"gather(): index must be a Tensor, got {type(index).__name__}")
    dim = _as_dim(bound["dim"])
    if old_order:
        warn_deprecated(
            _GATHER_OLD_ORDER,
            since=_GATHER_SINCE,
            removal=_GATHER_REMOVAL,
            alternative="gather(input, dim, index)",
        )
    return dim, cast("Tensor", index)


def _gather_adapter(input: _Impl, *args: object, **kwargs: object) -> _Impl:
    """gather(input, dim, index), read by :func:`_gather_operands`.

    The parameters after ``input`` stay open because the old order binds
    by type rather than by name: a signature naming either order would
    turn the other one's calls away before they got here.  ``input`` is
    named so that ``input=`` binds; the free function unwraps positional
    tensors only, so a keyword one arrives as a Tensor — hence ``_unwrap``.
    """
    # The reference order with positional ints, as every caller inside
    # Lucid writes it (cross_entropy and nll_loss among them), skips the
    # parsing below.
    if (
        not kwargs
        and len(args) == 2
        and type(args[0]) is int
        and hasattr(args[1], "_impl")
    ):
        return _C_engine.gather(
            _unwrap(input), _unwrap(cast("Tensor", args[1])), args[0]
        )
    dim, index = _gather_operands(args, kwargs)
    return _C_engine.gather(_unwrap(input), _unwrap(index), dim)


# ── Composite indexing adapters ──────────────────────────────────────────────


def _take_adapter(a_impl: _Impl, indices: Tensor) -> _Impl:
    """take(a, indices) — ``a`` read as 1-D, in the shape of ``indices``.

    The engine op takes a 1-D index only, so any other shape — a matrix of
    positions, a 0-d one — raised ``ShapeMismatch`` where the reference
    answers in the index's shape.
    """
    idx = _unwrap(indices)
    shape = list(idx.shape)
    if len(shape) == 1:
        return _C_engine.take(a_impl, idx)
    flat = _C_engine.take(a_impl, _C_engine.reshape(idx, [-1]))
    return _C_engine.reshape(flat, shape)


def _index_select_adapter(a_impl: _Impl, dim: int, index: Tensor) -> _Impl:
    """index_select(a, dim, index) — third arg is a tensor."""
    return _C_engine.index_select(a_impl, int(dim), _unwrap(index))


#: ``scatter``'s ``reduce=`` modes, each its own engine kernel.
_SCATTER_REDUCE = {"add": _C_engine.scatter_add, "multiply": _C_engine.scatter_prod}


def _scatter_adapter(
    x_impl: _Impl,
    dim: int,
    index: Tensor,
    src: Tensor | Scalar,
    reduce: str | None = None,
) -> _Impl:
    """scatter(base, dim, index, src, reduce=None).

    ``src`` may be a Python number, written at every position the index
    names.  A tensor ``src`` must have ``base``'s dtype: the engine's
    overwrite scatter casts it, as an assignment inside Lucid wants, but
    the reference refuses here as it does for every scatter — and as the
    engine's ``reduce`` kernels already do.  The index goes over at full
    width, so one past the axis is refused rather than wrapped.
    """
    idx_impl = _unwrap(index)
    if isinstance(src, _promotion.NUMBER):
        src_impl = _C_engine.full(
            list(idx_impl.shape), src, x_impl.dtype, x_impl.device
        )
    else:
        src_impl = _unwrap(src)
        if src_impl.dtype != x_impl.dtype:
            raise _C_engine.DtypeMismatch(
                f"scatter: src must have self's dtype "
                f"{_ENGINE_TO_DTYPE[x_impl.dtype]}, got "
                f"{_ENGINE_TO_DTYPE[src_impl.dtype]} — cast it first"
            )
    if reduce is None:
        return _C_engine.scatter(x_impl, int(dim), idx_impl, src_impl)
    kernel = _SCATTER_REDUCE.get(reduce)
    if kernel is None:
        raise ValueError(f"scatter: reduce must be 'add' or 'multiply', got {reduce!r}")
    return kernel(x_impl, idx_impl, src_impl, int(dim))


def _broadcast_with_mask(op: str, a_impl: _Impl, mask: _Impl) -> tuple[_Impl, _Impl]:
    """``a_impl`` and ``mask`` broadcast to their common shape, as the reference's are.

    The engine's masked kernels walk the two buffers side by side, so each
    operand has to arrive at the shape the other one sees.
    """
    a_shape = list(a_impl.shape)
    m_shape = list(mask.shape)
    rank = max(len(a_shape), len(m_shape))
    a_full = [1] * (rank - len(a_shape)) + a_shape
    m_full = [1] * (rank - len(m_shape)) + m_shape
    shape: list[int] = []
    for i, (x, y) in enumerate(zip(a_full, m_full, strict=True)):
        if x != y and 1 not in (x, y):
            raise ValueError(
                f"{op}: input shape {tuple(a_shape)} and mask shape "
                f"{tuple(m_shape)} do not broadcast (dimension {i}: {x} vs {y})"
            )
        shape.append(max(x, y))
    if a_shape != shape:
        a_impl = _C_engine.broadcast_to(a_impl, shape)
    if m_shape != shape:
        mask = _C_engine.broadcast_to(mask, shape)
    return a_impl, mask


def _masked_select_adapter(a_impl: _Impl, mask: _Impl) -> _Impl:
    """masked_select(a, mask) — ``a``'s elements where ``mask`` holds, as 1-D.

    Both operands broadcast to their common shape first, as the reference
    framework's do.  The engine kernel did not: given a mask of a different
    shape it walked the two buffers side by side, so a row mask returned
    the first row's picks only and a broadcast input read past its end.
    The kernel also records no graph, so an input that tracks gradients
    goes through boolean indexing instead, which scatters the gradient
    back to the selected positions.
    """
    a_impl, mask = _broadcast_with_mask("masked_select", a_impl, mask)
    if mask.dtype != _C_engine.Bool:
        mask = _C_engine.astype(mask, _C_engine.Bool)
    if a_impl.requires_grad and _C_engine.grad_enabled():
        from lucid._dispatch import _wrap

        return _unwrap(_wrap(a_impl)[_wrap(mask)])
    return _C_engine.masked_select(
        _C_engine.contiguous(a_impl), _C_engine.contiguous(mask)
    )


def _sort_adapter(a_impl: _Impl, dim: int = -1, descending: bool = False) -> _Impl:
    """sort(a, dim=-1, descending=False) — the sorted values along ``dim``."""
    out = _C_engine.sort(a_impl, int(dim))
    if descending:
        out = _C_engine.flip(out, [int(dim)])
    return out


def _argsort_adapter(a_impl: _Impl, dim: int = -1, descending: bool = False) -> _Impl:
    """argsort(a, dim=-1, descending=False).

    Descending keeps equal keys in their input order, as ascending does:
    sort the reversed input ascending, map each position ``p`` of it back
    to index ``n - 1 - p``, and reverse.  Reversing the ascending indices
    instead would put ties last-first.
    """
    d = int(dim)
    if not descending:
        return _C_engine.argsort(a_impl, d)
    reversed_order = _C_engine.argsort(_C_engine.flip(a_impl, [d]), d)
    last = _C_engine.full_like(reversed_order, float(a_impl.shape[d] - 1))
    return _C_engine.flip(_C_engine.sub(last, reversed_order), [d])


def _topk_adapter(
    a_impl: _Impl, k: int, dim: int = -1, largest: bool = True
) -> tuple[_Impl, _Impl]:
    """topk(a, k, dim=-1, largest=True) — ``(values, indices)``, best first.

    The smallest ``k`` are the head of the stable ascending order, so equal
    keys keep their input order and NaN, which sorts last, is taken only
    when fewer than ``k`` numbers remain.
    """
    d = int(dim)
    if largest:
        values, indices = _C_engine.topk(a_impl, int(k), d)
        return values, indices
    order = _C_engine.narrow(_C_engine.argsort(a_impl, d), d, 0, int(k))
    return _C_engine.gather(a_impl, order, d), order


def _chunk_adapter(a_impl: _Impl, chunks: int, dim: int = 0) -> list[_Impl]:
    """chunk(a, chunks, dim=0).

    Pieces of ``ceil(n / chunks)`` along ``dim``, the last one shorter —
    so fewer than ``chunks`` pieces come back when that size leaves nothing
    for the tail (7 split into 4 is 2, 2, 2, 1; 6 split into 4 is 2, 2, 2).
    """
    d = int(dim)
    count = int(chunks)
    if count <= 0:
        raise ValueError(f"chunk: chunks must be positive, got {count}")
    size = int(a_impl.shape[d])
    if size % count == 0:
        return _C_engine.chunk(a_impl, count, d)
    step = (size + count - 1) // count
    return _C_engine.split_at(a_impl, list(range(step, size, step)), d)


def _swapaxes_adapter(a_impl: _Impl, axis0: int, axis1: int) -> _Impl:
    """swapaxes(a, axis0, axis1) — the free function's parameter names."""
    return _C_engine.swapaxes(a_impl, int(axis0), int(axis1))


def _kthvalue_adapter(
    a_impl: _Impl,
    k: int,
    dim: int = -1,
    keepdim: bool = False,
) -> _Impl:
    """kthvalue(a, k, dim=-1, keepdim=False)."""
    return _C_engine.kthvalue(a_impl, int(k), int(dim), bool(keepdim))


def _narrow_adapter(a_impl: _Impl, dim: int, start: int, length: int) -> _Impl:
    """narrow(a, dim, start, length)."""
    return _own_impl(
        _C_engine.narrow(a_impl, int(dim), int(start), int(length)), a_impl
    )


def _as_strided_adapter(
    x_impl: _Impl,
    size: Sequence[int],
    stride: Sequence[int],
    storage_offset: int | None = None,
) -> _Impl:
    """as_strided(x, size, stride, storage_offset=None), all in elements."""
    return _C_engine.as_strided(
        x_impl,
        [int(s) for s in size],
        [int(s) for s in stride],
        None if storage_offset is None else int(storage_offset),
    )


# ── Layout / shape adapters ──────────────────────────────────────────────────


def _movedim_adapter(
    a_impl: _Impl,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> _Impl:
    """movedim(a, source, destination) — accept int or list for either arg."""
    src = [int(source)] if isinstance(source, int) else [int(s) for s in source]
    dst = (
        [int(destination)]
        if isinstance(destination, int)
        else [int(d) for d in destination]
    )
    return _C_engine.movedim(a_impl, src, dst)


def _unflatten_adapter(a_impl: _Impl, dim: int, sizes: Sequence[int]) -> _Impl:
    """unflatten(a, dim, sizes)."""
    return _C_engine.unflatten(a_impl, int(dim), [int(s) for s in sizes])


def _view_adapter(a_impl: _Impl, *shape: int | Sequence[int]) -> _Impl:
    """view(a, *shape) — accept ``view(t, 2, 3)`` and ``view(t, [2, 3])``."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        s = [int(d) for d in shape[0]]
    else:
        s = [int(d) for d in shape]  # type: ignore[arg-type]
    return _C_engine.view(a_impl, s)


def _concat_adapter(tensors: Sequence[Tensor], dim: int = 0) -> _Impl:
    """concat(tensors, dim=0) — first arg is a list of tensors."""
    return _C_engine.concat(_promote_impls([_unwrap(t) for t in tensors]), int(dim))


def _reshape_adapter(x_impl: _Impl, *shape: int | Sequence[int]) -> _Impl:
    """reshape(x, *shape) — accept variadic ints or single list/tuple."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        s = [int(d) for d in shape[0]]
    elif len(shape) == 1 and isinstance(shape[0], int):
        s = [int(shape[0])]
    else:
        s = [int(d) for d in shape]  # type: ignore[arg-type]
    return _C_engine.reshape(x_impl, s)


def _transpose_adapter(
    x_impl: _Impl, dim0: int | None = None, dim1: int | None = None
) -> _Impl:
    """transpose(x) swaps the last two axes; transpose(x, dim0, dim1) swaps those.

    The two-axis form is the reference framework's, and code written for it
    raised a TypeError here — the engine op takes no axes.
    """
    if dim0 is None and dim1 is None:
        return _C_engine.transpose(x_impl)
    if dim0 is None or dim1 is None:
        raise TypeError("transpose takes no axes, or two: transpose(dim0, dim1)")
    return _C_engine.swapaxes(x_impl, int(dim0), int(dim1))


def _permute_adapter(x_impl: _Impl, *dims: int | Sequence[int]) -> _Impl:
    """permute(x, *dims) — accept variadic ints or single list/tuple."""
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
        p = [int(d) for d in dims[0]]
    else:
        p = [int(d) for d in dims]  # type: ignore[arg-type]
    return _C_engine.permute(x_impl, p)


def _expand_adapter(x_impl: _Impl, *sizes: int | Sequence[int]) -> _Impl:
    """expand(x, *sizes) — accept variadic ints or single list/tuple.

    ``-1`` in any position means *keep the existing size along that dim*,
    matching the reference framework semantics.
    """
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        raw = [int(d) for d in sizes[0]]
    else:
        raw = [int(d) for d in sizes]  # type: ignore[arg-type]
    # Resolve -1 entries: replace with the corresponding source dimension.
    src_shape = list(x_impl.shape)
    ndim_src = len(src_shape)
    ndim_dst = len(raw)
    # If expanding to more dims, prepend 1s to src_shape (implicit broadcast).
    if ndim_dst > ndim_src:
        src_shape = [1] * (ndim_dst - ndim_src) + src_shape
    resolved = [src_shape[i] if d == -1 else d for i, d in enumerate(raw)]
    return _C_engine.expand(x_impl, resolved)


def _flip_adapter(x_impl: _Impl, dims: int | Sequence[int]) -> _Impl:
    """flip(x, dims) — accept ``dims=int`` or list/tuple of ints."""
    dims_list = [int(dims)] if isinstance(dims, int) else [int(d) for d in dims]
    return _C_engine.flip(x_impl, dims_list)


def _fliplr_adapter(x_impl: _Impl) -> _Impl:
    """fliplr(x) — flip along axis 1.  ``x`` must be at least 2-D."""
    if len(x_impl.shape) < 2:
        raise ValueError(
            f"fliplr: input must be at least 2-D, got shape {tuple(x_impl.shape)}"
        )
    return _C_engine.flip(x_impl, [1])


def _flipud_adapter(x_impl: _Impl) -> _Impl:
    """flipud(x) — flip along axis 0.  ``x`` must be at least 1-D."""
    if len(x_impl.shape) < 1:
        raise ValueError(
            f"flipud: input must be at least 1-D, got shape {tuple(x_impl.shape)}"
        )
    return _C_engine.flip(x_impl, [0])


# ── Parameters the reference spells differently ─────────────────────────────
#
# ``tril(x, diagonal=1)``, ``flatten(x, start_dim=1)`` and ``tile(x,
# dims=...)`` reached the engine binding, whose parameters are ``k``,
# ``start_axis`` / ``end_axis`` and ``reps``, and failed there with a raw
# "incompatible function arguments".  The released Lucid names hold their
# position — renaming a parameter a caller may pass by keyword breaks that
# caller — so the reference's spelling is added beside each, by keyword.


def _either_spelling[T](
    op: str, lucid_name: str, lucid_value: T | None, ref_name: str, ref_value: T | None
) -> T | None:
    """The one value given for a parameter that has two names, or ``None``."""
    if ref_value is None:
        return lucid_value
    if lucid_value is not None:
        raise TypeError(
            f"{op}() got values for both {lucid_name!r} and {ref_name!r}, "
            f"which name the same argument; pass one"
        )
    return ref_value


def _tril_adapter(
    a_impl: _Impl, k: int | None = None, *, diagonal: int | None = None
) -> _Impl:
    """tril(input, k=0, *, diagonal=None) — ``diagonal`` is the reference's ``k``."""
    offset = _either_spelling("tril", "k", k, "diagonal", diagonal)
    return _C_engine.tril(a_impl, 0 if offset is None else offset)


def _triu_adapter(
    a_impl: _Impl, k: int | None = None, *, diagonal: int | None = None
) -> _Impl:
    """triu(input, k=0, *, diagonal=None) — ``diagonal`` is the reference's ``k``."""
    offset = _either_spelling("triu", "k", k, "diagonal", diagonal)
    return _C_engine.triu(a_impl, 0 if offset is None else offset)


def _flatten_adapter(
    a_impl: _Impl,
    start_axis: int | None = None,
    end_axis: int | None = None,
    *,
    start_dim: int | None = None,
    end_dim: int | None = None,
) -> _Impl:
    """flatten(input, start_axis=0, end_axis=-1, *, start_dim, end_dim)."""
    start = _either_spelling(
        "flatten", "start_axis", start_axis, "start_dim", start_dim
    )
    end = _either_spelling("flatten", "end_axis", end_axis, "end_dim", end_dim)
    return _C_engine.flatten(
        a_impl, 0 if start is None else start, -1 if end is None else end
    )


def _counts(value: int | Sequence[int] | None) -> list[int] | None:
    """``value`` as per-axis counts: one int is one count."""
    if value is None:
        return None
    return [value] if isinstance(value, int) else [int(r) for r in value]


def _tile_adapter(
    a_impl: _Impl,
    reps: int | Sequence[int] | None = None,
    *,
    dims: int | Sequence[int] | None = None,
) -> _Impl:
    """tile(input, reps, *, dims) — ``dims`` is the reference's ``reps``.

    A single int is one count, as ``x.tile(2)`` reads in the reference.
    Fewer counts than axes are padded with leading 1s, as documented and as
    the reference does; the engine refuses them.
    """
    counts = _either_spelling("tile", "reps", _counts(reps), "dims", _counts(dims))
    if counts is None:
        raise TypeError("tile() missing required argument 'dims'")
    short = len(a_impl.shape) - len(counts)
    return _C_engine.tile(a_impl, [1] * short + counts)


def _squeeze_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
) -> _Impl:
    """squeeze(x, dim=None) — None drops all size-1; list squeezes multiple.

    A dim that is not size 1 is left alone, as the reference does — and the
    result is still a view of its own (see :func:`_own_impl`).
    """
    if dim is None:
        return _own_impl(_C_engine.squeeze_all(x_impl), x_impl)
    if isinstance(dim, (list, tuple)):
        ndim = len(x_impl.shape)
        result = x_impl
        # Sort descending so each ``squeeze`` keeps the remaining indices valid.
        for d in sorted([int(d) for d in dim], reverse=True):
            nd = d if d >= 0 else ndim + d
            if 0 <= nd < ndim and int(x_impl.shape[nd]) == 1:
                result = _C_engine.squeeze(result, nd)
                ndim -= 1
        return _own_impl(result, x_impl)
    ndim = len(x_impl.shape)
    d = int(cast(int, dim))
    nd = d if d >= 0 else ndim + d
    # Silently no-op on non-unit dim (matches reference behaviour).
    if nd < 0 or nd >= ndim or int(x_impl.shape[nd]) != 1:
        return _own_impl(x_impl, x_impl)
    return _C_engine.squeeze(x_impl, nd)


# ── Reduction adapters (dim/keepdim/correction kwargs) ───────────────────────


def _sum_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """sum(x, dim=None, keepdim=False)."""
    return _C_engine.sum(x_impl, _reduction_axes(x_impl, dim, "sum"), bool(keepdim))


def _mean_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """mean(x, dim=None, keepdim=False)."""
    return _C_engine.mean(x_impl, _reduction_axes(x_impl, dim, "mean"), bool(keepdim))


def _prod_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """prod(x, dim=None, keepdim=False)."""
    return _C_engine.prod(x_impl, _reduction_axes(x_impl, dim, "prod"), bool(keepdim))


def _max_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """max(x, dim=None, keepdim=False)."""
    return _C_engine.max(x_impl, _reduction_axes(x_impl, dim, "max"), bool(keepdim))


def _min_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """min(x, dim=None, keepdim=False)."""
    return _C_engine.min(x_impl, _reduction_axes(x_impl, dim, "min"), bool(keepdim))


def _flat_arg_reduce(
    engine_fn: Callable[[_Impl, int, bool], _Impl],
    name: str,
    x_impl: _Impl,
    keepdim: bool,
) -> _Impl:
    """``dim=None``: the index into the flattened tensor, as documented.

    It used to reduce the last axis instead, so ``x.argmax()`` of a matrix
    answered one index per row where the reference — and the docstring —
    give one into the whole tensor.  ``keepdim`` keeps every axis, at 1.
    """
    shape = list(x_impl.shape)
    if math.prod(shape) == 0:
        raise IndexError(
            f"{name}(): Expected reduction dim to be specified for input.numel() == 0."
        )
    out = engine_fn(_C_engine.reshape(x_impl, [-1]), 0, False)
    return _C_engine.reshape(out, [1] * len(shape)) if keepdim else out


def _argmax_adapter(
    x_impl: _Impl,
    dim: int | None = None,
    keepdim: bool = False,
) -> _Impl:
    """argmax(x, dim=None, keepdim=False)."""
    if dim is None:
        return _flat_arg_reduce(_C_engine.argmax, "argmax", x_impl, bool(keepdim))
    return _C_engine.argmax(x_impl, int(dim), bool(keepdim))


def _argmin_adapter(
    x_impl: _Impl,
    dim: int | None = None,
    keepdim: bool = False,
) -> _Impl:
    """argmin(x, dim=None, keepdim=False)."""
    if dim is None:
        return _flat_arg_reduce(_C_engine.argmin, "argmin", x_impl, bool(keepdim))
    return _C_engine.argmin(x_impl, int(dim), bool(keepdim))


def _truth_count(a_impl: _Impl, axes: list[int], keepdim: bool, falsy: bool) -> _Impl:
    """How many entries along ``axes`` are non-zero (``falsy``: are zero).

    A sum rather than ``max`` / ``min`` of the truth values: those refuse an
    empty reduced axis, where ``any`` is ``False`` and ``all`` ``True``.  A
    complex entry is non-zero when either part is; Metal's complex-to-bool
    cast reads the real part alone.
    """
    if a_impl.dtype in (_C_engine.C64, _C_engine.C128):
        truth = _C_engine.logical_or(
            _C_engine.astype(_C_engine.real(a_impl), _C_engine.Bool),
            _C_engine.astype(_C_engine.imag(a_impl), _C_engine.Bool),
        )
    else:
        truth = _C_engine.astype(a_impl, _C_engine.Bool)
    if falsy:
        truth = _C_engine.logical_not(truth)
    return _C_engine.sum(truth, axes, keepdim)


def _logical_reduce(
    name: str, a_impl: _Impl, dim: int | Sequence[int] | None, keepdim: bool
) -> _Impl:
    """``any`` / ``all`` over ``dim`` — ``all`` is "no entry is zero"."""
    if dim is None and not keepdim:
        return _C_engine.any(a_impl) if name == "any" else _C_engine.all(a_impl)
    axes = _reduction_axes(a_impl, dim, name)
    counted = _truth_count(a_impl, axes, bool(keepdim), falsy=name == "all")
    none = _C_engine.equal(counted, _C_engine.zeros_like(counted))
    return none if name == "all" else _C_engine.logical_not(none)


def _any_adapter(
    a_impl: _Impl, dim: int | Sequence[int] | None = None, keepdim: bool = False
) -> _Impl:
    """any(input, dim=None, keepdim=False) — whether some entry is non-zero."""
    return _logical_reduce("any", a_impl, dim, keepdim)


def _all_adapter(
    a_impl: _Impl, dim: int | Sequence[int] | None = None, keepdim: bool = False
) -> _Impl:
    """all(input, dim=None, keepdim=False) — whether every entry is non-zero."""
    return _logical_reduce("all", a_impl, dim, keepdim)


def _logsumexp_adapter(
    a_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """logsumexp(a, dim=None, keepdim=False) — accept axis/None like sum/mean."""
    axes = _reduction_axes(a_impl, dim, "logsumexp")
    # An integer or bool input is summed in the default float dtype, as the
    # reference framework does — the engine op computes in the input dtype.
    if _promotion.is_integral(a_impl.dtype):
        a_impl = _C_engine.astype(a_impl, to_engine_dtype(None))
    return _C_engine.logsumexp(a_impl, axes, bool(keepdim))


# ── Repeat / split adapters ──────────────────────────────────────────────────


def _repeat_adapter(x_impl: _Impl, repeats: int, dim: int | None = None) -> _Impl:
    """``lucid.repeat(x, repeats, dim=None)`` — interleaved replication along
    axis 0 when ``dim`` is None, else along the given axis.

    Distinct from ``Tensor.repeat`` (below): the free function follows
    NumPy-style ``repeat`` semantics, while the method tiles like the
    reference framework's ``Tensor.repeat``.
    """
    axis = 0 if dim is None else int(dim)
    return _C_engine.repeat(x_impl, int(repeats), axis)


def _repeat_method_adapter(x_impl: _Impl, *sizes: int | Sequence[int]) -> _Impl:
    """``Tensor.repeat(*sizes)`` — tile copies along each dim.

    Routes to ``engine.tile`` so the method semantics match the reference
    framework's ``Tensor.repeat`` (and stay separated from the free
    function above).
    """
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        reps = list(sizes[0])
    else:
        reps = list(sizes)
    return _C_engine.tile(x_impl, [int(r) for r in reps])


def _repeat_interleave_adapter(
    a_impl: _Impl,
    repeats: int,
    dim: int | None = None,
) -> _Impl:
    """repeat_interleave(x, repeats, dim=None) — defers to engine.repeat.

    The engine's ``repeat`` op already implements interleaved replication
    along a single axis; ``dim=None`` means flatten first, then repeat
    along axis 0.
    """
    if dim is None:
        flat = _C_engine.reshape(a_impl, [int(a_impl.numel())])
        return _C_engine.repeat(flat, int(repeats), 0)
    return _C_engine.repeat(a_impl, int(repeats), int(dim))


def _split_adapter(
    x_impl: _Impl,
    split_size_or_sections: int | Sequence[int],
    dim: int = 0,
) -> list[_Impl]:
    """split(x, sections, dim=0) — int → pieces of that size, the last one
    shorter when it does not divide; list → explicit sizes."""
    axis_size = int(x_impl.shape[dim])
    if isinstance(split_size_or_sections, int):
        size = split_size_or_sections
        if size <= 0:
            raise ValueError(f"split: split_size must be positive, got {size}")
        if axis_size and axis_size % size == 0:
            return _C_engine.split(x_impl, axis_size // size, int(dim))
        return _C_engine.split_at(x_impl, list(range(size, axis_size, size)), int(dim))
    indices: list[int] = []
    cumsum = 0
    for s in split_size_or_sections[:-1]:
        cumsum += int(s)
        indices.append(cumsum)
    return _C_engine.split_at(x_impl, indices, int(dim))


# ── tensordot / meshgrid / where / masked_fill / pad ─────────────────────────


def _tensordot_adapter(
    a_impl: _Impl,
    b_impl: _Impl,
    dims: int | Sequence[int] | Sequence[Sequence[int]] = 2,
    _axes_b: Sequence[int] | None = None,
) -> _Impl:
    """tensordot — accept int / nested-list / pair-of-lists ``dims``."""
    if _axes_b is not None:
        axes_a = [int(d) for d in cast(Sequence[int], dims)]
        axes_b = [int(d) for d in _axes_b]
    elif isinstance(dims, int):
        ra = len(a_impl.shape)
        axes_a = list(range(ra - int(dims), ra))
        axes_b = list(range(int(dims)))
    else:
        _dims_seq = cast(Sequence[Sequence[int]], dims)
        axes_a = [int(d) for d in _dims_seq[0]]
        axes_b = [int(d) for d in _dims_seq[1]]
    return _C_engine.tensordot(a_impl, b_impl, axes_a, axes_b)


def _outer_adapter(a_impl: _Impl, b_impl: _Impl) -> _Impl:
    """outer(a, b) — the two vectors at their common dtype.

    The engine multiplies operands of one dtype only, so an ``int64`` vector
    beside a ``float32`` one raised ``DtypeMismatch``; the reference answers
    in ``float32``, as ``a * b`` would.  Shared by ``Tensor.outer`` and
    ``lucid.linalg.outer`` so the two cannot drift.

    Both callers hand over impls already, and they are not unwrapped again:
    shadow allocation's phantom impls pass one ``_unwrap`` but not a second.
    A Python number is refused rather than coerced, as the bare engine op
    refused it, since an outer product with one is not an outer product.
    """
    for operand in (a_impl, b_impl):
        if isinstance(operand, (bool, int, float, complex)):
            raise TypeError(f"outer: expected a tensor, got {type(operand).__name__}")
    a_impl, b_impl = _promote_impls([a_impl, b_impl])
    return _C_engine.outer(a_impl, b_impl)


_keep_engine_signature(_outer_adapter, _C_engine.outer)


def _meshgrid_adapter(*tensors: Tensor, indexing: str = "ij") -> list[_Impl]:
    """meshgrid(*tensors, indexing='ij') — variadic input; ``indexing`` kwarg
    selects between matrix ('ij') and Cartesian ('xy') ordering."""
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    impls = [_unwrap(t) for t in tensors]
    return _C_engine.meshgrid(impls, indexing == "xy")


def _clip_adapter(
    x_impl: _Impl,
    min: Scalar | None = None,
    max: Scalar | None = None,
) -> _Impl:
    """clip(x, min, max) — either bound may be omitted.

    The engine primitive wants both ends, so a one-sided clamp used to
    raise a pybind argument error rather than doing the obvious thing.
    Substituting an infinity would have been wrong for integer dtypes,
    where it has no representable value; routing a single bound through
    ``maximum`` / ``minimum`` instead needs no stand-in value at all.

    A bound promotes the result as it would in ``x + bound``, so a float
    bound on an integer tensor answers in float, as the reference's does.
    Taking the tensor's dtype instead truncated ``clamp(ints, 0.5, 2.5)`` to
    whole numbers and made the one-sided form raise ``DtypeMismatch``.  A
    Python number is weak here as everywhere: it can change the kind but
    never the width, so a ``float16`` tensor stays ``float16``.
    """
    if min is None and max is None:
        raise ValueError("clip: at least one of min or max must be given")
    # A number is promoted as it stands, without building its 0-d constant,
    # so a clamp whose bounds already fit allocates nothing extra.
    bounds: list[_promotion.Operand] = []
    for bound in (min, max):
        if bound is None:
            continue
        if isinstance(bound, _promotion.NUMBER):
            bounds.append(bound)
        else:  # a tensor bound, which the annotation does not admit but callers pass
            bounds.append(_unwrap(bound))
    tgt = _promotion.result_type(x_impl, *bounds)
    if tgt != x_impl.dtype:
        x_impl = _C_engine.astype(x_impl, tgt)
    if min is None or max is None:
        bound_impl = _unwrap_or_scalar(max if min is None else min, x_impl)
        if bound_impl.dtype != tgt:
            bound_impl = _C_engine.astype(bound_impl, tgt)
        one_sided = _C_engine.minimum if min is None else _C_engine.maximum
        return one_sided(x_impl, bound_impl)
    return _C_engine.clip(x_impl, float(min), float(max))


def _where_adapter(
    cond: Tensor,
    x: TensorOrScalar | None = None,
    y: TensorOrScalar | None = None,
) -> _Impl | tuple[_Impl, ...]:
    """where(cond, x, y) — bool-cast the condition, promote the branches.

    ``where(cond)`` alone is ``nonzero(cond, as_tuple=True)``, as in the
    reference: the indices of the true elements, one tensor per dim.

    Either branch may be a Python scalar, which is the common spelling for
    a masked constant: ``where(x > 0, 1.0, -1.0)``.  Each is promoted
    against whichever operand is already a tensor, so the pair keeps the
    dtype it would have had written the long way.

    The two branches then meet at their common dtype, as in ``x + y``: the
    engine selects between tensors of one dtype only, so an ``int64``
    branch beside a ``float32`` one — or beside ``1.5`` — raised
    ``DtypeMismatch`` where the reference answers in float.
    """
    if x is None and y is None:
        return _nonzero_adapter(_unwrap(cond), as_tuple=True)
    if x is None or y is None:
        raise TypeError("where() takes either cond alone, or cond, x and y")
    c = _unwrap(cond)
    if c.dtype != _C_engine.Bool:
        c = _C_engine.astype(c, _C_engine.Bool)
    # Tensor is a TYPE_CHECKING-only name here, so the test is for
    # scalar-ness rather than for the wrapper type.
    ref_impl = c
    for branch in (x, y):
        if not isinstance(branch, (bool, int, float, complex)):
            ref_impl = _unwrap(branch)
            break
    x_impl, y_impl = _promote_impls(
        [_unwrap_or_scalar(x, ref_impl), _unwrap_or_scalar(y, ref_impl)]
    )
    return _C_engine.where(c, x_impl, y_impl)


def _nonzero_adapter(
    a_impl: _Impl, *, as_tuple: bool = False
) -> _Impl | tuple[_Impl, ...]:
    """nonzero(a, *, as_tuple=False) — the indices of ``a``'s nonzero elements.

    ``as_tuple`` returns one 1-D index tensor per dim rather than one
    ``[count, ndim]`` matrix; a 0-d input counts as 1-D there, as in the
    reference.  How many elements are nonzero is read back to size the
    result — the one host round trip this op needs.
    """
    if not as_tuple:
        return _C_engine.nonzero(a_impl)
    if not a_impl.shape:
        a_impl = _C_engine.reshape(a_impl, [1])
    return tuple(_C_engine.unbind(_C_engine.nonzero(a_impl), 1))


def _masked_fill_adapter(x_impl: _Impl, mask: Tensor, value: float) -> _Impl:
    """masked_fill(x, mask, value) — auto-cast mask to bool.

    ``x`` and the mask broadcast to their common shape, as the reference's
    do: the engine kernel reads them side by side, so a ``[2, 1, 1, 4]``,
    ``[4]`` or 0-d mask against a ``[2, 3, 4, 4]`` input raised
    ``ShapeMismatch`` instead.

    An integer or bool tensor has no infinity and no NaN, and the engine's
    cast wrote whatever the conversion gave: ``-inf`` became ``INT64_MIN``
    with nothing to say so, which then reads as a huge negative number
    rather than a mask.  The reference refuses such a value, and so does
    this; a finite one is cast to the tensor's dtype as before.
    """
    m = _unwrap(mask)
    if m.dtype != _C_engine.Bool:
        m = _C_engine.astype(m, _C_engine.Bool)
    x_impl, m = _broadcast_with_mask("masked_fill", x_impl, m)
    fill = float(value)
    if not math.isfinite(fill) and _promotion.is_integral(x_impl.dtype):
        dtype = _ENGINE_TO_DTYPE.get(x_impl.dtype, x_impl.dtype)
        raise RuntimeError(
            f"masked_fill: {fill} cannot be written into a {dtype} tensor, "
            "which has no infinity or NaN — cast it to a floating dtype first"
        )
    return _C_engine.masked_fill(x_impl, m, fill)


def _pad_adapter(
    x_impl: _Impl,
    padding: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
) -> _Impl:
    """pad(x, padding, mode, value) — reference flat (last-dim-first) convention."""
    if mode != "constant":
        raise NotImplementedError(
            f"pad: mode={mode!r} not supported; only 'constant' is wired"
        )
    ndim = len(x_impl.shape)
    n_pad_dims = len(padding) // 2
    pad_pairs: list[tuple[int, int]] = [(0, 0)] * ndim
    for i in range(n_pad_dims):
        dim_idx = ndim - 1 - i
        pad_pairs[dim_idx] = (int(padding[2 * i]), int(padding[2 * i + 1]))
    return _C_engine.pad(x_impl, pad_pairs, float(value))


# ── Stats / search / combinatorial ───────────────────────────────────────────


def _histc_adapter(
    a_impl: _Impl,
    bins: int = 100,
    min: float = 0.0,
    max: float = 0.0,
) -> _Impl:
    """histc(a, bins=100, min=0, max=0) — counts over ``[min, max]``, as the reference.

    ``min == max`` takes the range from the data; a range that is still a
    single point widens to ``[v - 1, v + 1]``, which puts ``v`` in the
    middle bin — the engine widened it to ``[v, v + 1]``, putting every
    element in the first.  A NaN lies in no bin: the kernel's bin index for
    one came out as 0, so it was counted there.  It is sent below the range
    instead, where elements are skipped.
    """
    lo, hi = float(min), float(max)
    if lo > hi:
        raise ValueError(f"histc: max must be at least min, got [{lo}, {hi}]")
    if lo == hi and a_impl.numel() > 0:
        # The range is the data's: two scalars read back, as the engine did.
        lo = float(_C_engine.min(a_impl, [], False).item())
        hi = float(_C_engine.max(a_impl, [], False).item())
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"histc: range of [{lo}, {hi}] is not finite")
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    if a_impl.dtype in _REAL_FLOATS:
        a_impl = _C_engine.nan_to_num(a_impl, -math.inf, math.inf, -math.inf)
    return _C_engine.histc(a_impl, int(bins), lo, hi)


#: The dtypes the engine's own ``unique`` kernel answers exactly: integers
#: have no NaN and no ``-0.0``, which its float path gets wrong, and it
#: sorts several times faster than the composite below on the CPU.
_UNIQUE_KERNEL_DTYPES = (_C_engine.I32, _C_engine.I64)


def _unique_adapter(
    a_impl: _Impl,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> _Impl | tuple[_Impl, ...]:
    """unique(a, sorted=True, return_inverse=False, return_counts=False, dim=None).

    The values always come back sorted, which the reference allows for
    ``sorted=False`` too.  ``dim=None`` reads ``a`` as 1-D and the inverse
    takes ``a``'s shape; with ``dim`` the unique values are whole slices
    along it and the inverse has one entry per slice.  How many there are
    is read back to size the result — the one host round trip.
    """
    del sorted  # sorted either way; the reference leaves the order open
    plain = dim is None and not (return_inverse or return_counts)
    if plain and a_impl.dtype in _UNIQUE_KERNEL_DTYPES:
        return _C_engine.unique(_C_engine.reshape(a_impl, [-1]))
    d: int | None = None  # the dim the slices run along; None reads ``a`` as 1-D
    rest: list[int] = []  # one slice's shape
    if dim is None:
        shape: list[int] = list(a_impl.shape)
        rows = _C_engine.reshape(a_impl, [a_impl.numel(), 1])
    else:
        ndim = len(a_impl.shape)
        if not -ndim <= dim < ndim:
            raise IndexError(f"unique: dim {dim} is out of range for a {ndim}-d tensor")
        d = dim % ndim
        moved = _C_engine.contiguous(_C_engine.movedim(a_impl, [d], [0]))
        rest = [int(s) for s in moved.shape[1:]]
        shape = [int(moved.shape[0])]
        # The width spelled out: ``-1`` cannot be inferred with no rows.
        rows = _C_engine.reshape(moved, [shape[0], math.prod(rest)])
    values, inverse, counts = _unique_rows(rows, return_inverse, return_counts)
    if d is None:
        values = _C_engine.reshape(values, [-1])
    else:
        values = _C_engine.reshape(values, [int(values.shape[0]), *rest])
        values = _C_engine.movedim(values, [0], [d])
    out = [values]
    if inverse is not None:
        out.append(_C_engine.reshape(inverse, shape))
    if counts is not None:
        out.append(counts)
    return out[0] if len(out) == 1 else tuple(out)


def _unique_rows(
    rows: _Impl, inverse: bool, counts: bool
) -> tuple[_Impl, _Impl | None, _Impl | None]:
    """The distinct rows of a 2-D ``rows``, sorted; each row's group; each group's size.

    Rows are ordered lexicographically by a stable sort per column, last
    column first; a single column is one sort, and needs no order at all
    when nobody asks where each row went.  Neighbouring rows that differ
    anywhere start a group, so NaN, which equals nothing, is a group of its
    own each time, and ``-0.0`` joins ``0.0``.  The group and size of each
    are computed only when asked for.
    """
    n, width = int(rows.shape[0]), int(rows.shape[1])
    device = rows.device
    order: _Impl | None = None
    if inverse or width != 1:
        order = _lexicographic_order(rows)
        ranked = _C_engine.index_select(rows, 0, order)
    else:
        ranked = _C_engine.sort(rows, 0)
    if n == 0:
        empty = _C_engine.zeros([0], _C_engine.I64, device)
        return ranked, empty if inverse else None, empty if counts else None
    differs = _C_engine.not_equal(
        _C_engine.narrow(ranked, 0, 1, n - 1), _C_engine.narrow(ranked, 0, 0, n - 1)
    )
    if width != 1:
        differs = _C_engine.greater(
            _C_engine.sum(_C_engine.astype(differs, _C_engine.I64), [1], False),
            _C_engine.zeros([n - 1], _C_engine.I64, device),
        )
    starts = _C_engine.concatenate(
        [_C_engine.ones([1], _C_engine.Bool, device), _C_engine.reshape(differs, [-1])],
        0,
    )
    first = _C_engine.reshape(_C_engine.nonzero(starts), [-1])
    groups: _Impl | None = None
    if order is not None and inverse:
        group = _C_engine.sub(
            _C_engine.cumsum(_C_engine.astype(starts, _C_engine.I64), 0),
            _C_engine.ones([n], _C_engine.I64, device),
        )
        zeros = _C_engine.zeros([n], _C_engine.I64, device)
        groups = _C_engine.scatter(zeros, 0, order, group)
    sizes: _Impl | None = None
    if counts:
        ends = _C_engine.concatenate(
            [
                _C_engine.narrow(first, 0, 1, int(first.shape[0]) - 1),
                _C_engine.full([1], n, _C_engine.I64, device),
            ],
            0,
        )
        sizes = _C_engine.sub(ends, first)
    return _C_engine.index_select(ranked, 0, first), groups, sizes


def _lexicographic_order(rows: _Impl) -> _Impl:
    """The ``int64`` permutation that sorts ``rows`` lexicographically, stably."""
    n, width = int(rows.shape[0]), int(rows.shape[1])
    if width == 1:
        step = _C_engine.argsort(_C_engine.reshape(rows, [n]), 0)
        return _C_engine.astype(step, _C_engine.I64)
    order = _C_engine.arange(0, n, 1, _C_engine.I64, rows.device)
    for col in range(width - 1, -1, -1):
        key = _C_engine.index_select(
            _C_engine.reshape(_C_engine.narrow(rows, 1, col, 1), [n]), 0, order
        )
        step = _C_engine.astype(_C_engine.argsort(key, 0), _C_engine.I64)
        order = _C_engine.index_select(order, 0, step)
    return order


def _cartesian_prod_adapter(*tensors: Tensor) -> _Impl:
    """cartesian_prod(*tensors) — accept variadic tensor args.

    The registry's list-arg path requires the user to pass a list;
    ``cartesian_prod(t1, t2)`` is the more natural calling form, so we
    register with ``n_tensor_args=0`` and unwrap each operand here.
    """
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    return _C_engine.cartesian_prod([_unwrap(t) for t in tensors])


def _searchsorted_adapter(
    sorted_seq: Tensor,
    values: Tensor,
    *,
    right: bool = False,
) -> _Impl:
    """searchsorted(sorted_seq, values, right=False) — accept tensor inputs."""
    return _C_engine.searchsorted(_unwrap(sorted_seq), _unwrap(values), bool(right))


def _bucketize_adapter(
    values: Tensor,
    boundaries: Tensor,
    *,
    right: bool = False,
) -> _Impl:
    """bucketize(values, boundaries, right=False)."""
    return _C_engine.bucketize(_unwrap(values), _unwrap(boundaries), bool(right))


def _isclose_adapter(
    a_impl: _Impl | Scalar,
    b_impl: _Impl | Scalar,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> _Impl:
    """isclose(a, b, rtol, atol, equal_nan) — equal_nan branch handled here.

    Either operand may be a Python scalar; comparing a tensor against a
    literal tolerance target is the ordinary use.
    """
    ref = a_impl if isinstance(a_impl, _C_engine.TensorImpl) else None
    b_impl = _unwrap_or_scalar(b_impl, ref)
    a_impl = _unwrap_or_scalar(a_impl, b_impl)
    if equal_nan:
        # ``isclose ∨ (isnan(a) ∧ isnan(b))`` — we lift to bitwise on bool
        # tensors so the result stays a single bool tensor.
        out = _C_engine.isclose(a_impl, b_impl, float(rtol), float(atol))
        nan_a = _C_engine.isnan(a_impl)
        nan_b = _C_engine.isnan(b_impl)
        both_nan = _C_engine.bitwise_and(nan_a, nan_b)
        return _C_engine.bitwise_or(out, both_nan)
    return _C_engine.isclose(a_impl, b_impl, float(rtol), float(atol))


# Sub-module forwarders for ``cross`` / ``norm`` / ``einsum`` were removed
# (2026-05-08).  Lucid's API tree exposes those ops only via their canonical
# sub-package paths — ``lucid.linalg.cross``, ``lucid.linalg.norm``,
# ``lucid.einops.einsum`` — so the adapters that used to forward them at the
# top level were dead code under the new policy.
