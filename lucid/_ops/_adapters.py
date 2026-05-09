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

Helpers ``_to_axes`` and ``_bessel_correct`` are shared across the reduction
adapters and live at the top of the module.
"""

from typing import Callable, Sequence, TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# Convenience alias for the engine's TensorImpl class — it's the lingua
# franca of every adapter signature.  Adapters consume the engine's raw
# storage type, while user-facing args are typed as ``Tensor``.
_Impl = _C_engine.TensorImpl


# ── Binary dtype-promotion helpers ───────────────────────────────────────────
# Mirrors the same table in ``_tensor/_dunders.py``.  Kept in sync manually;
# both live in Python-only infrastructure (no external deps, H4-safe).

_D = _C_engine.Dtype
_ARITH_DTYPE_KIND_WIDTH: dict[_C_engine.Dtype, tuple[int, int]] = {
    _D.Bool: (0, 1),
    _D.I8: (1, 8),
    _D.I16: (1, 16),
    _D.I32: (1, 32),
    _D.I64: (1, 64),
    _D.F16: (2, 16),
    _D.F32: (2, 32),
    _D.F64: (2, 64),
    _D.C64: (3, 64),
}


def _arith_result_dtype(da: _C_engine.Dtype, db: _C_engine.Dtype) -> _C_engine.Dtype:
    """Return the promoted dtype for an arithmetic binary op."""
    if da == db:
        return da
    ka, wa = _ARITH_DTYPE_KIND_WIDTH.get(da, (2, 32))
    kb, wb = _ARITH_DTYPE_KIND_WIDTH.get(db, (2, 32))
    if ka != kb:
        return da if ka > kb else db
    return da if wa >= wb else db


def _make_arith_adapter(
    engine_fn: Callable[[_Impl, _Impl], _Impl],
) -> Callable[[_Impl, _Impl], _Impl]:
    """Wrap an arithmetic binary engine function with automatic dtype promotion.

    When both operands have the same dtype the call is a zero-overhead
    passthrough.  Otherwise each operand is cast to the promoted dtype before
    forwarding to the engine — matching the type-promotion behaviour of the
    reference framework.

    Only arithmetic ops (add/sub/mul/div/pow/maximum/minimum) should use this
    wrapper.  matmul/dot/inner/outer deliberately bypass it because they have
    their own dtype constraints that the engine enforces.
    """

    def _adapter(a: _Impl, b: _Impl) -> _Impl:
        da, db = a.dtype, b.dtype
        if da != db:
            tgt = _arith_result_dtype(da, db)
            if da != tgt:
                a = _C_engine.astype(a, tgt)
            if db != tgt:
                b = _C_engine.astype(b, tgt)
        return engine_fn(a, b)

    # Preserve the pybind11 docstring (contains the canonical signature line)
    # so that gen_pyi.py's _parse_pybind_signature path extracts the correct
    # Tensor-typed signature rather than the internal _Impl-typed one.
    _adapter.__doc__ = getattr(engine_fn, "__doc__", None)
    _adapter.__name__ = getattr(engine_fn, "__name__", "_arith_adapter")
    # Store the original engine function so that _signature_for_entry can
    # fall back to pybind11 doc-string parsing when the AST path fails.
    _adapter.__wrapped__ = engine_fn  # type: ignore[attr-defined]
    return _adapter


# ── Shared helpers ───────────────────────────────────────────────────────────


def _to_axes(dim: int | Sequence[int] | None) -> list[int]:
    """Convert ``None | int | list[int]`` → ``list[int]`` for the engine."""
    if dim is None:
        return []
    if isinstance(dim, (list, tuple)):
        return [int(d) for d in dim]
    return [int(dim)]


def _bessel_correct(
    result_impl: _Impl,
    x_impl: _Impl,
    axes_list: Sequence[int],
    correction: int,
) -> _Impl:
    """Scale a ddof=0 variance to match ``correction``.  No-op when 0."""
    if correction == 0:
        return result_impl
    n = 1
    if axes_list:
        for ax in axes_list:
            n *= int(x_impl.shape[ax])
    else:
        for s in x_impl.shape:
            n *= int(s)
    if n <= correction:
        return result_impl
    scale = float(n) / float(n - correction)
    scale_t = _C_engine.full(
        list(result_impl.shape), scale, result_impl.dtype, result_impl.device
    )
    return _C_engine.mul(result_impl, scale_t)


# ── Engine-arg-order adapters ────────────────────────────────────────────────


def _detach_adapter(impl: _Impl) -> _Impl:
    """detach(x): deep-copy without gradient tracking."""
    return _C_engine.contiguous(impl).clone_with_grad(False)


def _scatter_add_adapter(
    x_impl: _Impl,
    dim: int,
    index: Tensor,
    src: Tensor,
) -> _Impl:
    """scatter_add(x, dim, index, src) — Python order → engine order.

    Engine takes ``(base, indices, src, dim)``; we reorder here.
    The engine scatter_add is buggy for int64 indices — coerce to int32.
    """
    idx_impl = _unwrap(index)
    if idx_impl.dtype == _C_engine.I64:
        idx_impl = _C_engine.astype(idx_impl, _C_engine.I32)
    return _C_engine.scatter_add(x_impl, idx_impl, _unwrap(src), dim)


# ── Composite indexing adapters ──────────────────────────────────────────────


def _take_adapter(a_impl: _Impl, indices: Tensor) -> _Impl:
    """take(a, indices) — second tensor needs unwrap."""
    return _C_engine.take(a_impl, _unwrap(indices))


def _index_select_adapter(a_impl: _Impl, dim: int, index: Tensor) -> _Impl:
    """index_select(a, dim, index) — third arg is a tensor."""
    return _C_engine.index_select(a_impl, int(dim), _unwrap(index))


def _scatter_adapter(
    base_impl: _Impl,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str | None = None,
) -> _Impl:
    """scatter(base, dim, index, src, reduce=None).

    ``reduce=None``  → overwrite (composite C++ op).
    ``reduce='add'`` → forward to ``scatter_add`` (separate engine kernel).
    Other reduce modes are not supported and raise ``NotImplementedError``.

    The engine's 1-D ``scatter_add`` path is buggy for int64 indices — coerce
    to int32 here so callers don't trip over it; the workaround stays
    invisible to user code.
    """
    idx_impl = _unwrap(index)
    if idx_impl.dtype == _C_engine.I64:
        idx_impl = _C_engine.astype(idx_impl, _C_engine.I32)
    if reduce is None:
        return _C_engine.scatter(base_impl, int(dim), idx_impl, _unwrap(src))
    if reduce == "add":
        return _C_engine.scatter_add(base_impl, idx_impl, _unwrap(src), int(dim))
    raise NotImplementedError(
        f"scatter reduce={reduce!r} is not implemented; use 'add' or None"
    )


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
    return _C_engine.narrow(a_impl, int(dim), int(start), int(length))


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
    return _C_engine.concat([_unwrap(t) for t in tensors], int(dim))


def _reshape_adapter(x_impl: _Impl, *shape: int | Sequence[int]) -> _Impl:
    """reshape(x, *shape) — accept variadic ints or single list/tuple."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        s = [int(d) for d in shape[0]]
    elif len(shape) == 1 and isinstance(shape[0], int):
        s = [int(shape[0])]
    else:
        s = [int(d) for d in shape]  # type: ignore[arg-type]
    return _C_engine.reshape(x_impl, s)


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


def _squeeze_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
) -> _Impl:
    """squeeze(x, dim=None) — None drops all size-1; list squeezes multiple."""
    if dim is None:
        return _C_engine.squeeze_all(x_impl)
    if isinstance(dim, (list, tuple)):
        ndim = len(x_impl.shape)
        result = x_impl
        # Sort descending so each ``squeeze`` keeps the remaining indices valid.
        for d in sorted([int(d) for d in dim], reverse=True):
            nd = d if d >= 0 else ndim + d
            if 0 <= nd < ndim and int(x_impl.shape[nd]) == 1:
                result = _C_engine.squeeze(result, nd)
                ndim -= 1
        return result
    ndim = len(x_impl.shape)
    d = int(dim)
    nd = d if d >= 0 else ndim + d
    # Silently no-op on non-unit dim (matches reference behaviour).
    if nd < 0 or nd >= ndim or int(x_impl.shape[nd]) != 1:
        return x_impl
    return _C_engine.squeeze(x_impl, nd)


# ── Reduction adapters (dim/keepdim/correction kwargs) ───────────────────────


def _sum_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """sum(x, dim=None, keepdim=False)."""
    return _C_engine.sum(x_impl, _to_axes(dim), bool(keepdim))


def _mean_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """mean(x, dim=None, keepdim=False)."""
    return _C_engine.mean(x_impl, _to_axes(dim), bool(keepdim))


def _prod_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """prod(x, dim=None, keepdim=False)."""
    return _C_engine.prod(x_impl, _to_axes(dim), bool(keepdim))


def _max_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """max(x, dim=None, keepdim=False)."""
    return _C_engine.max(x_impl, _to_axes(dim), bool(keepdim))


def _min_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """min(x, dim=None, keepdim=False)."""
    return _C_engine.min(x_impl, _to_axes(dim), bool(keepdim))


def _var_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    correction: int = 1,
    unbiased: bool | None = None,
) -> _Impl:
    """var(x, dim, keepdim, correction=1) — ddof default matches reference."""
    if unbiased is not None:
        correction = 1 if unbiased else 0
    ax = _to_axes(dim)
    result = _C_engine.var(x_impl, ax, bool(keepdim))
    return _bessel_correct(result, x_impl, ax, correction)


def _std_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    correction: int = 1,
    unbiased: bool | None = None,
) -> _Impl:
    """std(x, dim, keepdim, correction=1) = sqrt(var(...))."""
    if unbiased is not None:
        correction = 1 if unbiased else 0
    ax = _to_axes(dim)
    v = _C_engine.var(x_impl, ax, bool(keepdim))
    v = _bessel_correct(v, x_impl, ax, correction)
    return _C_engine.sqrt(v)


def _argmax_adapter(
    x_impl: _Impl,
    dim: int | None = None,
    keepdim: bool = False,
) -> _Impl:
    """argmax(x, dim=None, keepdim=False)."""
    return _C_engine.argmax(x_impl, -1 if dim is None else int(dim), bool(keepdim))


def _argmin_adapter(
    x_impl: _Impl,
    dim: int | None = None,
    keepdim: bool = False,
) -> _Impl:
    """argmin(x, dim=None, keepdim=False)."""
    return _C_engine.argmin(x_impl, -1 if dim is None else int(dim), bool(keepdim))


def _logsumexp_adapter(
    a_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """logsumexp(a, dim=None, keepdim=False) — accept axis/None like sum/mean."""
    if dim is None:
        axes: list[int] = []
    elif isinstance(dim, (list, tuple)):
        axes = [int(d) for d in dim]
    else:
        axes = [int(dim)]
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
        reps = list(sizes)  # type: ignore[arg-type]
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
    """split(x, sections, dim=0) — int → equal chunks; list → explicit sizes."""
    axis_size = int(x_impl.shape[dim])
    if isinstance(split_size_or_sections, int):
        chunk = split_size_or_sections
        n = (axis_size + chunk - 1) // chunk
        return _C_engine.split(x_impl, n, int(dim))
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
        axes_a = [int(d) for d in dims]  # type: ignore[union-attr]
        axes_b = [int(d) for d in _axes_b]
    elif isinstance(dims, int):
        ra = len(a_impl.shape)
        axes_a = list(range(ra - int(dims), ra))
        axes_b = list(range(int(dims)))
    else:
        axes_a = [int(d) for d in dims[0]]  # type: ignore[index]
        axes_b = [int(d) for d in dims[1]]  # type: ignore[index]
    return _C_engine.tensordot(a_impl, b_impl, axes_a, axes_b)


def _meshgrid_adapter(*tensors: Tensor, indexing: str = "ij") -> list[_Impl]:
    """meshgrid(*tensors, indexing='ij') — variadic input; ``indexing`` kwarg
    selects between matrix ('ij') and Cartesian ('xy') ordering."""
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    impls = [_unwrap(t) for t in tensors]
    return _C_engine.meshgrid(impls, indexing == "xy")


def _where_adapter(cond: Tensor, x: Tensor, y: Tensor) -> _Impl:
    """where(cond, x, y) — auto-cast cond to bool to match reference behaviour."""
    c = _unwrap(cond)
    if c.dtype != _C_engine.Bool:
        c = _C_engine.astype(c, _C_engine.Bool)
    return _C_engine.where(c, _unwrap(x), _unwrap(y))


def _masked_fill_adapter(x_impl: _Impl, mask: Tensor, value: float) -> _Impl:
    """masked_fill(x, mask, value) — auto-cast mask to bool."""
    m = _unwrap(mask)
    if m.dtype != _C_engine.Bool:
        m = _C_engine.astype(m, _C_engine.Bool)
    return _C_engine.masked_fill(x_impl, m, float(value))


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
    """histc(a, bins=100, min=0, max=0) — defaults match the reference framework."""
    return _C_engine.histc(a_impl, int(bins), float(min), float(max))


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
    a_impl: _Impl,
    b_impl: _Impl,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> _Impl:
    """isclose(a, b, rtol, atol, equal_nan) — equal_nan branch handled here."""
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
