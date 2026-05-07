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

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap

# ── Shared helpers ───────────────────────────────────────────────────────────


def _to_axes(dim):
    """Convert ``None | int | list[int]`` → ``list[int]`` for the engine."""
    if dim is None:
        return []
    if isinstance(dim, (list, tuple)):
        return [int(d) for d in dim]
    return [int(dim)]


def _bessel_correct(result_impl, x_impl, axes_list, correction):
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


def _detach_adapter(impl):
    """detach(x): deep-copy without gradient tracking."""
    return _C_engine.contiguous(impl).clone_with_grad(False)


def _scatter_add_adapter(x_impl, dim: int, index, src):
    """scatter_add(x, dim, index, src) — Python order → engine order.

    Engine takes ``(base, indices, src, dim)``; we reorder here.
    The engine scatter_add is buggy for int64 indices — coerce to int32.
    """
    idx_impl = _unwrap(index)
    if idx_impl.dtype == _C_engine.I64:
        idx_impl = _C_engine.astype(idx_impl, _C_engine.I32)
    return _C_engine.scatter_add(x_impl, idx_impl, _unwrap(src), dim)


# ── Composite indexing adapters ──────────────────────────────────────────────


def _take_adapter(a_impl, indices):
    """take(a, indices) — second tensor needs unwrap."""
    return _C_engine.take(a_impl, _unwrap(indices))


def _index_select_adapter(a_impl, dim: int, index):
    """index_select(a, dim, index) — third arg is a tensor."""
    return _C_engine.index_select(a_impl, int(dim), _unwrap(index))


def _scatter_adapter(base_impl, dim: int, index, src, reduce=None):
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


def _kthvalue_adapter(a_impl, k, dim=-1, keepdim=False):
    """kthvalue(a, k, dim=-1, keepdim=False)."""
    return _C_engine.kthvalue(a_impl, int(k), int(dim), bool(keepdim))


def _narrow_adapter(a_impl, dim, start, length):
    """narrow(a, dim, start, length)."""
    return _C_engine.narrow(a_impl, int(dim), int(start), int(length))


# ── Layout / shape adapters ──────────────────────────────────────────────────


def _movedim_adapter(a_impl, source, destination):
    """movedim(a, source, destination) — accept int or list for either arg."""
    src = [int(source)] if isinstance(source, int) else [int(s) for s in source]
    dst = (
        [int(destination)]
        if isinstance(destination, int)
        else [int(d) for d in destination]
    )
    return _C_engine.movedim(a_impl, src, dst)


def _unflatten_adapter(a_impl, dim, sizes):
    """unflatten(a, dim, sizes)."""
    return _C_engine.unflatten(a_impl, int(dim), [int(s) for s in sizes])


def _view_adapter(a_impl, *shape):
    """view(a, *shape) — accept ``view(t, 2, 3)`` and ``view(t, [2, 3])``."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        s = [int(d) for d in shape[0]]
    else:
        s = [int(d) for d in shape]
    return _C_engine.view(a_impl, s)


def _concat_adapter(tensors, dim: int = 0):
    """concat(tensors, dim=0) — first arg is a list of tensors."""
    return _C_engine.concat([_unwrap(t) for t in tensors], int(dim))


def _reshape_adapter(x_impl, *shape):
    """reshape(x, *shape) — accept variadic ints or single list/tuple."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        s = [int(d) for d in shape[0]]
    elif len(shape) == 1 and isinstance(shape[0], int):
        s = [int(shape[0])]
    else:
        s = [int(d) for d in shape]
    return _C_engine.reshape(x_impl, s)


def _permute_adapter(x_impl, *dims):
    """permute(x, *dims) — accept variadic ints or single list/tuple."""
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
        p = [int(d) for d in dims[0]]
    else:
        p = [int(d) for d in dims]
    return _C_engine.permute(x_impl, p)


def _expand_adapter(x_impl, *sizes):
    """expand(x, *sizes) — accept variadic ints or single list/tuple."""
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        s = [int(d) for d in sizes[0]]
    else:
        s = [int(d) for d in sizes]
    return _C_engine.expand(x_impl, s)


def _flip_adapter(x_impl, dims):
    """flip(x, dims) — accept ``dims=int`` or list/tuple of ints."""
    dims_list = [int(dims)] if isinstance(dims, int) else [int(d) for d in dims]
    return _C_engine.flip(x_impl, dims_list)


def _fliplr_adapter(x_impl):
    """fliplr(x) — flip along axis 1.  ``x`` must be at least 2-D."""
    if len(x_impl.shape) < 2:
        raise ValueError(
            f"fliplr: input must be at least 2-D, got shape {tuple(x_impl.shape)}"
        )
    return _C_engine.flip(x_impl, [1])


def _flipud_adapter(x_impl):
    """flipud(x) — flip along axis 0.  ``x`` must be at least 1-D."""
    if len(x_impl.shape) < 1:
        raise ValueError(
            f"flipud: input must be at least 1-D, got shape {tuple(x_impl.shape)}"
        )
    return _C_engine.flip(x_impl, [0])


def _squeeze_adapter(x_impl, dim=None):
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
    x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
):
    """sum(x, dim=None, keepdim=False) with axis/axes/keepdims aliases."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.sum(x_impl, ax, kd)


def _mean_adapter(
    x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
):
    """mean(x, dim=None, keepdim=False)."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.mean(x_impl, ax, kd)


def _prod_adapter(
    x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
):
    """prod(x, dim=None, keepdim=False)."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.prod(x_impl, ax, kd)


def _max_adapter(
    x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
):
    """max(x, dim=None, keepdim=False)."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.max(x_impl, ax, kd)


def _min_adapter(
    x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None
):
    """min(x, dim=None, keepdim=False)."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.min(x_impl, ax, kd)


def _var_adapter(
    x_impl,
    dim=None,
    keepdim=False,
    *,
    correction=1,
    unbiased=None,
    axis=None,
    axes=None,
    keepdims=None,
):
    """var(x, dim, keepdim, correction=1) — ddof default matches reference."""
    if unbiased is not None:
        correction = 1 if unbiased else 0
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    result = _C_engine.var(x_impl, ax, kd)
    return _bessel_correct(result, x_impl, ax, correction)


def _std_adapter(
    x_impl,
    dim=None,
    keepdim=False,
    *,
    correction=1,
    unbiased=None,
    axis=None,
    axes=None,
    keepdims=None,
):
    """std(x, dim, keepdim, correction=1) = sqrt(var(...))."""
    if unbiased is not None:
        correction = 1 if unbiased else 0
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    v = _C_engine.var(x_impl, ax, kd)
    v = _bessel_correct(v, x_impl, ax, correction)
    return _C_engine.sqrt(v)


def _argmax_adapter(x_impl, dim=None, keepdim=False, *, axis=None, keepdims=None):
    """argmax(x, dim=None, keepdim=False)."""
    d = dim if dim is not None else axis
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.argmax(x_impl, -1 if d is None else int(d), bool(kd))


def _argmin_adapter(x_impl, dim=None, keepdim=False, *, axis=None, keepdims=None):
    """argmin(x, dim=None, keepdim=False)."""
    d = dim if dim is not None else axis
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.argmin(x_impl, -1 if d is None else int(d), bool(kd))


def _logsumexp_adapter(a_impl, dim=None, keepdim=False):
    """logsumexp(a, dim=None, keepdim=False) — accept axis/None like sum/mean."""
    if dim is None:
        axes = []
    elif isinstance(dim, (list, tuple)):
        axes = [int(d) for d in dim]
    else:
        axes = [int(dim)]
    return _C_engine.logsumexp(a_impl, axes, bool(keepdim))


# ── Repeat / split adapters ──────────────────────────────────────────────────


def _repeat_adapter(x_impl, repeats, dim=None):
    """``lucid.repeat(x, repeats, dim=None)`` — interleaved replication along
    axis 0 when ``dim`` is None, else along the given axis.

    Distinct from ``Tensor.repeat`` (below): the free function follows
    NumPy-style ``repeat`` semantics, while the method tiles like the
    reference framework's ``Tensor.repeat``.
    """
    axis = 0 if dim is None else int(dim)
    return _C_engine.repeat(x_impl, int(repeats), axis)


def _repeat_method_adapter(x_impl, *sizes):
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


def _repeat_interleave_adapter(a_impl, repeats, dim=None):
    """repeat_interleave(x, repeats, dim=None) — defers to engine.repeat.

    The engine's ``repeat`` op already implements interleaved replication
    along a single axis; ``dim=None`` means flatten first, then repeat
    along axis 0.
    """
    if dim is None:
        flat = _C_engine.reshape(a_impl, [int(a_impl.numel())])
        return _C_engine.repeat(flat, int(repeats), 0)
    return _C_engine.repeat(a_impl, int(repeats), int(dim))


def _split_adapter(x_impl, split_size_or_sections, dim=0):
    """split(x, sections, dim=0) — int → equal chunks; list → explicit sizes."""
    axis_size = int(x_impl.shape[dim])
    if isinstance(split_size_or_sections, int):
        chunk = split_size_or_sections
        n = (axis_size + chunk - 1) // chunk
        return _C_engine.split(x_impl, n, int(dim))
    indices = []
    cumsum = 0
    for s in split_size_or_sections[:-1]:
        cumsum += s
        indices.append(cumsum)
    return _C_engine.split_at(x_impl, indices, int(dim))


# ── tensordot / meshgrid / where / masked_fill / pad ─────────────────────────


def _tensordot_adapter(a_impl, b_impl, dims=2, _axes_b=None):
    """tensordot — accept int / nested-list / pair-of-lists ``dims``."""
    if _axes_b is not None:
        axes_a = [int(d) for d in dims]
        axes_b = [int(d) for d in _axes_b]
    elif isinstance(dims, int):
        ra = len(a_impl.shape)
        axes_a = list(range(ra - int(dims), ra))
        axes_b = list(range(int(dims)))
    else:
        axes_a = [int(d) for d in dims[0]]
        axes_b = [int(d) for d in dims[1]]
    return _C_engine.tensordot(a_impl, b_impl, axes_a, axes_b)


def _meshgrid_adapter(*tensors, indexing="ij"):
    """meshgrid(*tensors, indexing='ij') — variadic input; ``indexing`` kwarg
    selects between matrix ('ij') and Cartesian ('xy') ordering."""
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    impls = [_unwrap(t) for t in tensors]
    return _C_engine.meshgrid(impls, indexing == "xy")


def _where_adapter(cond, x, y):
    """where(cond, x, y) — auto-cast cond to bool to match reference behaviour."""
    c = _unwrap(cond)
    if c.dtype != _C_engine.Bool:
        c = _C_engine.astype(c, _C_engine.Bool)
    return _C_engine.where(c, _unwrap(x), _unwrap(y))


def _masked_fill_adapter(x_impl, mask, value):
    """masked_fill(x, mask, value) — auto-cast mask to bool."""
    m = _unwrap(mask)
    if m.dtype != _C_engine.Bool:
        m = _C_engine.astype(m, _C_engine.Bool)
    return _C_engine.masked_fill(x_impl, m, float(value))


def _pad_adapter(x_impl, padding, mode="constant", value=0.0):
    """pad(x, padding, mode, value) — reference flat (last-dim-first) convention."""
    if mode != "constant":
        raise NotImplementedError(
            f"pad: mode={mode!r} not supported; only 'constant' is wired"
        )
    ndim = len(x_impl.shape)
    n_pad_dims = len(padding) // 2
    pad_pairs = [(0, 0)] * ndim
    for i in range(n_pad_dims):
        dim_idx = ndim - 1 - i
        pad_pairs[dim_idx] = (int(padding[2 * i]), int(padding[2 * i + 1]))
    return _C_engine.pad(x_impl, pad_pairs, float(value))


# ── Stats / search / combinatorial ───────────────────────────────────────────


def _histc_adapter(a_impl, bins=100, min=0.0, max=0.0):
    """histc(a, bins=100, min=0, max=0) — defaults match the reference framework."""
    return _C_engine.histc(a_impl, int(bins), float(min), float(max))


def _cartesian_prod_adapter(*tensors):
    """cartesian_prod(*tensors) — accept variadic tensor args.

    The registry's list-arg path requires the user to pass a list;
    ``cartesian_prod(t1, t2)`` is the more natural calling form, so we
    register with ``n_tensor_args=0`` and unwrap each operand here.
    """
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    return _C_engine.cartesian_prod([_unwrap(t) for t in tensors])


def _searchsorted_adapter(sorted_seq, values, *, right=False):
    """searchsorted(sorted_seq, values, right=False) — accept tensor inputs."""
    return _C_engine.searchsorted(_unwrap(sorted_seq), _unwrap(values), bool(right))


def _bucketize_adapter(values, boundaries, *, right=False):
    """bucketize(values, boundaries, right=False)."""
    return _C_engine.bucketize(_unwrap(values), _unwrap(boundaries), bool(right))


def _isclose_adapter(a_impl, b_impl, rtol=1e-5, atol=1e-8, equal_nan=False):
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


# ── Sub-module forwarders ────────────────────────────────────────────────────


def _cross_adapter(a_impl, b_impl, dim: int = -1):
    """cross(a, b, dim=-1) — forwards to lucid.linalg.cross.

    ``lucid.linalg.cross`` is a Python wrapper around an engine kernel; this
    adapter lets the top-level ``lucid.cross`` behave identically without
    duplicating the wrapper logic.
    """
    from lucid._dispatch import _wrap as _w
    from lucid.linalg import cross as _linalg_cross

    return _unwrap(_linalg_cross(_w(a_impl), _w(b_impl), dim=int(dim)))


def _norm_adapter(a_impl, ord=None, dim=None, keepdim=False):
    """norm(a, ord, dim, keepdim) — forwards to lucid.linalg.norm."""
    from lucid._dispatch import _wrap as _w
    from lucid.linalg import norm as _linalg_norm

    return _unwrap(_linalg_norm(_w(a_impl), ord=ord, dim=dim, keepdim=keepdim))


def _einsum_adapter(equation, *operands):
    """einsum(equation, *operands) — top-level alias for ``lucid.einops.einsum``.

    The engine's einsum kernel lives in the ``einops`` sub-module; we forward
    here so the top-level ``lucid.einsum(...)`` calling form works without
    bypassing the registry.
    """
    impls = [_unwrap(t) if hasattr(t, "_impl") else t for t in operands]
    return _C_engine.einops.einsum(equation, impls)
