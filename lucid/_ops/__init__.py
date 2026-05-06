"""
Free-function ops exposed as lucid.xxx.

All functions are generated from the registry. Pure-Python implementations
for missing engine ops (std, log_softmax, any, all) are also defined here.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._registry import _REGISTRY

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _make_free_fn(name: str) -> object:
    """Create a free function that wraps an engine function."""
    for entry in _REGISTRY:
        fn_name = entry.free_fn_name or entry.name
        if fn_name == name:
            e = entry

            if e.n_tensor_args == -1:

                def _fn_list(tensors: list[Tensor], *args: object) -> object:
                    impls = [_unwrap(t) for t in tensors]
                    result = e.engine_fn(impls, *args)
                    if e.returns_tensor:
                        if isinstance(result, (list, tuple)):
                            return type(result)(_wrap(r) for r in result)
                        return _wrap(result)
                    return result

                _fn_list.__name__ = fn_name
                return _fn_list
            else:

                def _fn(*args: object, **kwargs: object) -> object:
                    proc: list[object] = []
                    for i, a in enumerate(args):
                        if i < e.n_tensor_args and hasattr(a, "_impl"):
                            proc.append(_unwrap(a))
                        else:
                            proc.append(a)
                    result = e.engine_fn(*proc, **kwargs)
                    if e.returns_tensor:
                        if isinstance(result, (list, tuple)):
                            return type(result)(_wrap(r) for r in result)
                        return _wrap(result)
                    return result

                _fn.__name__ = fn_name
                return _fn
    raise AttributeError(f"No op found for free function: {name}")


# ── generate all free functions from registry ─────────────────────────────

_FREE_FN_NAMES = set()


def _populate_free_fns() -> None:
    for entry in _REGISTRY:
        fn_name = entry.free_fn_name
        if fn_name is None:
            continue
        if fn_name in _FREE_FN_NAMES:
            continue
        _FREE_FN_NAMES.add(fn_name)
        globals()[fn_name] = _make_free_fn(fn_name)


_populate_free_fns()


# ── API-compatible overrides ──────────────────────────────────────────────
# These replace the auto-generated functions above with wrappers that accept
# API-compatible keyword arguments (dim, keepdim, correction, dims, etc.).


def _to_axes(dim):
    """Convert dim/axis (None | int | list[int]) → list[int] for the engine."""
    if dim is None:
        return []
    if isinstance(dim, (list, tuple)):
        return [int(d) for d in dim]
    return [int(dim)]


def _bessel_correct(result_impl, x_impl, axes_list, correction):
    """Scale a ddof=0 variance by n/(n-correction) in-place (returns new impl)."""
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


# ── reductions ────────────────────────────────────────────────────────────────


def sum(x, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """Sum; accepts API-compatible dim/keepdim."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _wrap(_C_engine.sum(_unwrap(x), ax, kd))


def mean(x, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """Mean; accepts API-compatible dim/keepdim."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _wrap(_C_engine.mean(_unwrap(x), ax, kd))


def prod(x, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """Product; accepts API-compatible dim/keepdim."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _wrap(_C_engine.prod(_unwrap(x), ax, kd))


def max(x, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """Max; accepts API-compatible dim/keepdim."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _wrap(_C_engine.max(_unwrap(x), ax, kd))


def min(x, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """Min; accepts API-compatible dim/keepdim."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _wrap(_C_engine.min(_unwrap(x), ax, kd))


def var(
    x,
    dim=None,
    keepdim=False,
    *,
    correction=1,
    unbiased=None,
    axis=None,
    axes=None,
    keepdims=None,
):
    """Variance; correction=1 applies Bessel's correction (reference default)."""
    if unbiased is not None:
        correction = 1 if unbiased else 0
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    impl = _unwrap(x)
    result = _C_engine.var(impl, ax, kd)
    return _wrap(_bessel_correct(result, impl, ax, correction))


def std(
    x,
    dim=None,
    keepdim=False,
    *,
    correction=1,
    unbiased=None,
    axis=None,
    axes=None,
    keepdims=None,
):
    """Std dev; correction=1 applies Bessel's correction (reference default)."""
    if unbiased is not None:
        correction = 1 if unbiased else 0
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    impl = _unwrap(x)
    v = _C_engine.var(impl, ax, kd)
    v = _bessel_correct(v, impl, ax, correction)
    return _wrap(_C_engine.sqrt(v))


def argmax(x, dim=None, keepdim=False, *, axis=None, keepdims=None):
    """Argmax; accepts API-compatible dim/keepdim (axis also accepted)."""
    d = dim if dim is not None else axis
    kd = keepdims if keepdims is not None else keepdim
    if d is None:
        return _wrap(_C_engine.argmax(_unwrap(x), -1, kd))
    return _wrap(_C_engine.argmax(_unwrap(x), int(d), kd))


def argmin(x, dim=None, keepdim=False, *, axis=None, keepdims=None):
    """Argmin; accepts API-compatible dim/keepdim (axis also accepted)."""
    d = dim if dim is not None else axis
    kd = keepdims if keepdims is not None else keepdim
    if d is None:
        return _wrap(_C_engine.argmin(_unwrap(x), -1, kd))
    return _wrap(_C_engine.argmin(_unwrap(x), int(d), kd))


# ── shape ─────────────────────────────────────────────────────────────────────


def squeeze(x, dim=None):
    """Squeeze; dim=None removes all size-1 dims; list squeezes multiple dims.
    Non-unit dims are silently ignored (reference behaviour).
    """
    impl = _unwrap(x)
    if dim is None:
        return _wrap(_C_engine.squeeze_all(impl))
    if isinstance(dim, (list, tuple)):
        ndim = len(impl.shape)
        result = impl
        for d in sorted([int(d) for d in dim], reverse=True):
            nd = d if d >= 0 else ndim + d
            if 0 <= nd < ndim and int(impl.shape[nd]) == 1:
                result = _C_engine.squeeze(result, nd)
                ndim -= 1
        return _wrap(result)
    ndim = len(impl.shape)
    d = int(dim)
    nd = d if d >= 0 else ndim + d
    if nd < 0 or nd >= ndim or int(impl.shape[nd]) != 1:
        return _wrap(impl)
    return _wrap(_C_engine.squeeze(impl, nd))


# ── repeat / tile ─────────────────────────────────────────────────────────────


def repeat(x, repeats, dim=None):
    """Alias for repeat_interleave: repeats elements along dim (NumPy / reference semantics).

    Note: matches repeat_interleave, NOT Tensor.repeat (which tiles copies).
    Use lucid.tile(x, reps) to tile copies in each dimension.
    """
    impl = _unwrap(x)
    axis = 0 if dim is None else int(dim)
    return _wrap(_C_engine.repeat(impl, int(repeats), axis))


def repeat_interleave(x, repeats, dim=None):
    """Repeat elements of a tensor (interleave, matches repeat_interleave)."""
    impl = _unwrap(x)
    axis = 0 if dim is None else int(dim)
    return _wrap(_C_engine.repeat(impl, int(repeats), axis))


# ── split ─────────────────────────────────────────────────────────────────────


def split(x, split_size_or_sections, dim=0):
    """Split a tensor into chunks (reference semantics: split_size = chunk size).

    Args:
        x:                    Input tensor.
        split_size_or_sections: int → chunk size; list[int] → explicit section sizes.
        dim:                  Dimension along which to split.
    """
    impl = _unwrap(x)
    axis_size = int(impl.shape[dim])
    if isinstance(split_size_or_sections, int):
        chunk_size = split_size_or_sections
        n = (axis_size + chunk_size - 1) // chunk_size
        parts = _C_engine.split(impl, n, int(dim))
    else:
        indices: list[int] = []
        cumsum = 0
        for s in split_size_or_sections[:-1]:
            cumsum += s
            indices.append(cumsum)
        parts = _C_engine.split_at(impl, indices, int(dim))
    return [_wrap(p) for p in parts]


# ── tensordot ─────────────────────────────────────────────────────────────────


def tensordot(a, b, dims=2, _axes_b=None):
    """Tensordot with API-compatible dims argument.

    Accepted call forms:
      tensordot(a, b, dims=2)               → contract last 2 of a with first 2 of b
      tensordot(a, b, dims=[[0,1],[2,3]])   → explicit axes lists (reference framework style)
      tensordot(a, b, [0,1], [2,3])         → separate positional axes (legacy style)
    """
    ai = _unwrap(a)
    bi = _unwrap(b)
    if _axes_b is not None:
        # Called as tensordot(a, b, axes_a, axes_b) — legacy 4-arg form
        axes_a = [int(d) for d in dims]
        axes_b = [int(d) for d in _axes_b]
    elif isinstance(dims, int):
        n = dims
        ra = len(ai.shape)
        axes_a = list(range(ra - n, ra))
        axes_b = list(range(n))
    else:
        axes_a = [int(d) for d in dims[0]]
        axes_b = [int(d) for d in dims[1]]
    return _wrap(_C_engine.tensordot(ai, bi, axes_a, axes_b))


# ── meshgrid ──────────────────────────────────────────────────────────────────


def meshgrid(*tensors, indexing="ij"):
    """Return coordinate matrices from coordinate vectors.

    indexing="ij" (default) → matrix indexing (like NumPy).
    indexing="xy"           → Cartesian indexing (x=cols, y=rows).
    """
    impls = [_unwrap(t) for t in tensors]
    indexing_xy = indexing == "xy"
    results = _C_engine.meshgrid(impls, indexing_xy)
    return [_wrap(r) for r in results]


# ── where / masked_fill — auto-cast condition to bool ─────────────────────────


def where(condition, x, y):
    """Select elements from x (where True) or y (where False).
    Condition is cast to bool dtype if needed (mirrors reference behaviour).
    """
    c = _unwrap(condition)
    if c.dtype != _C_engine.Bool:
        c = _C_engine.astype(c, _C_engine.Bool)
    return _wrap(_C_engine.where(c, _unwrap(x), _unwrap(y)))


def masked_fill(x, mask, value):
    """Fill positions where mask is True with value.
    Mask is cast to bool dtype if needed.
    """
    impl = _unwrap(x)
    m = _unwrap(mask)
    if m.dtype != _C_engine.Bool:
        m = _C_engine.astype(m, _C_engine.Bool)
    return _wrap(_C_engine.masked_fill(impl, m, float(value)))


# ── pad — reference framework flat format → engine per-dim pairs ─────────────────────────


def pad(x, padding, mode="constant", value=0.0):
    """Pad tensor using reference framework flat convention (last-dim first).
    e.g. padding=(1, 1, 2, 2) pads last-dim by (1,1) and second-to-last by (2,2).
    """
    impl = _unwrap(x)
    ndim = len(impl.shape)
    n_pad_dims = len(padding) // 2
    pad_pairs = [(0, 0)] * ndim
    for i in range(n_pad_dims):
        dim_idx = ndim - 1 - i
        pad_pairs[dim_idx] = (int(padding[2 * i]), int(padding[2 * i + 1]))
    return _wrap(_C_engine.pad(impl, pad_pairs, float(value)))


# ── reshape / permute / expand — varargs → list ───────────────────────────────


def reshape(x, *shape):
    """Reshape tensor. Accepts (t, new_shape) or (t, d0, d1, ...) forms."""
    impl = _unwrap(x)
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        s = [int(d) for d in shape[0]]
    elif len(shape) == 1 and isinstance(shape[0], int):
        s = [int(shape[0])]
    else:
        s = [int(d) for d in shape]
    return _wrap(_C_engine.reshape(impl, s))


def permute(x, *dims):
    """Permute axes. Accepts (t, perm_list) or (t, d0, d1, ...) forms."""
    impl = _unwrap(x)
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
        p = [int(d) for d in dims[0]]
    else:
        p = [int(d) for d in dims]
    return _wrap(_C_engine.permute(impl, p))


def expand(x, *sizes):
    """Expand tensor to shape. Accepts (t, sizes_list) or (t, s0, s1, ...) forms."""
    impl = _unwrap(x)
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        s = [int(d) for d in sizes[0]]
    else:
        s = [int(d) for d in sizes]
    return _wrap(_C_engine.expand(impl, s))


# ── register all overrides in globals and __all__ ─────────────────────────────

# ── Shape / View aliases ──────────────────────────────────────────────────────


def view(x, *shape):
    """Reshape alias matching the standard tensor framework's ``Tensor.view``.

    Accepts either a single tuple/list of sizes or a variadic int form, and
    forwards verbatim to ``reshape``. ``view`` is an alias kept for parity —
    Lucid does not distinguish view/copy semantics at this layer.
    """
    return reshape(x, *shape)


def concat(tensors, dim=0):
    """Alias for ``cat`` — concatenate a sequence of tensors along ``dim``."""
    impls = [_unwrap(t) for t in tensors]
    return _wrap(_C_engine.concatenate(impls, int(dim)))


def narrow(x, dim, start, length):
    """Return a view sliced along ``dim`` to ``[start, start + length)``.

    Equivalent to the standard ``Tensor.narrow``; implemented with the
    existing slice op so autograd is preserved.
    """
    impl = _unwrap(x)
    ndim = len(impl.shape)
    d = int(dim) + ndim if int(dim) < 0 else int(dim)
    if d < 0 or d >= ndim:
        raise IndexError(f"narrow: dim {dim} out of range for ndim={ndim}")
    s = int(start)
    n = int(length)
    if s < 0 or n < 0 or s + n > int(impl.shape[d]):
        raise IndexError(
            f"narrow: range [{s}, {s + n}) out of bounds for size {int(impl.shape[d])}"
        )
    idx = [slice(None)] * ndim
    idx[d] = slice(s, s + n)
    from lucid._tensor.tensor import Tensor as _T

    return _T(impl)[tuple(idx)]


def movedim(x, source, destination):
    """Move tensor dimensions from ``source`` positions to ``destination``.

    Both arguments may be a single int or a list/tuple of ints; the result
    is a permutation of the input's axes such that each ``source[i]`` ends
    up at ``destination[i]``.
    """
    impl = _unwrap(x)
    ndim = len(impl.shape)
    src = [int(source)] if isinstance(source, int) else [int(s) for s in source]
    dst = [int(destination)] if isinstance(destination, int) else [int(d) for d in destination]
    if len(src) != len(dst):
        raise ValueError("movedim: source and destination must have same length")
    src = [s + ndim if s < 0 else s for s in src]
    dst = [d + ndim if d < 0 else d for d in dst]
    if len(set(src)) != len(src) or len(set(dst)) != len(dst):
        raise ValueError("movedim: each source/destination must be unique")
    rest = [d for d in range(ndim) if d not in src]
    perm = [0] * ndim
    for s, d in zip(src, dst):
        perm[d] = s
    rest_iter = iter(rest)
    for d in range(ndim):
        if perm[d] == 0 and d not in dst:
            perm[d] = next(rest_iter)
    # Fix the placeholder zeros: the algorithm above can collide on perm[d]==0
    # when zero is a legitimate source — recompute cleanly.
    perm = [-1] * ndim
    for s, d in zip(src, dst):
        perm[d] = s
    rest_iter = iter(rest)
    for d in range(ndim):
        if perm[d] == -1:
            perm[d] = next(rest_iter)
    return permute(x, perm)


def unflatten(x, dim, sizes):
    """Inverse of ``flatten`` — split ``dim`` into the given ``sizes``."""
    impl = _unwrap(x)
    ndim = len(impl.shape)
    d = int(dim) + ndim if int(dim) < 0 else int(dim)
    if d < 0 or d >= ndim:
        raise IndexError(f"unflatten: dim {dim} out of range for ndim={ndim}")
    sizes_list = [int(s) for s in sizes]
    new_shape = list(impl.shape[:d]) + sizes_list + list(impl.shape[d + 1 :])
    return reshape(x, new_shape)


# ── Indexing extras ───────────────────────────────────────────────────────────


def take(x, indices):
    """Return a 1-D tensor of values at ``indices`` from a flattened ``x``."""
    impl = _unwrap(x)
    flat = _C_engine.reshape(impl, [int(impl.numel())])
    idx = _unwrap(indices)
    if idx.dtype not in (_C_engine.I32, _C_engine.I64):
        raise TypeError("take: indices must be int32 or int64")
    return _wrap(_C_engine.gather(flat, idx, 0))


def index_select(x, dim, index):
    """Select rows/cols of ``x`` along ``dim`` according to ``index``."""
    impl = _unwrap(x)
    idx = _unwrap(index)
    if idx.dtype not in (_C_engine.I32, _C_engine.I64):
        raise TypeError("index_select: index must be int32 or int64")
    ndim = len(impl.shape)
    d = int(dim) + ndim if int(dim) < 0 else int(dim)
    # Broadcast idx to the source shape with size = len(idx) on dim ``d``;
    # gather expects index to have the same rank as the source tensor.
    target_shape = list(impl.shape)
    target_shape[d] = int(idx.shape[0])
    expanded_idx_shape = [1] * ndim
    expanded_idx_shape[d] = int(idx.shape[0])
    idx_reshaped = _C_engine.reshape(idx, expanded_idx_shape)
    idx_full = _C_engine.expand(idx_reshaped, target_shape)
    return _wrap(_C_engine.gather(impl, idx_full, d))


def masked_select(x, mask):
    """Return a 1-D tensor of elements where ``mask`` is True."""
    return _wrap(_C_engine.masked_select(_unwrap(x), _unwrap(mask)))


def scatter(x, dim, index, src, reduce=None):
    """Out-of-place scatter — write ``src`` values into ``x`` at ``index``.

    Currently supports the no-reduce form (overwrite) and ``reduce="add"``.
    Other reductions raise ``NotImplementedError``.
    """
    impl = _unwrap(x)
    idx = _unwrap(index)
    # The engine's 1-D scatter_add path is buggy for int64 indices — coerce to
    # int32 here (and document the workaround) so callers don't trip over it.
    if idx.dtype == _C_engine.I64:
        idx = _C_engine.astype(idx, _C_engine.I32)
    if reduce is None:
        # ``out[idx[i]] = src[i]`` — emulate via gather/scatter_add by computing
        # the delta needed to overwrite each scattered slot.
        srci = _unwrap(src)
        d = int(dim)
        existing = _C_engine.gather(impl, idx, d)
        delta = _C_engine.sub(srci, existing)
        return _wrap(_C_engine.scatter_add(impl, idx, delta, d))
    if reduce == "add":
        return _wrap(
            _C_engine.scatter_add(impl, idx, _unwrap(src), int(dim))
        )
    raise NotImplementedError(
        f"scatter reduce={reduce!r} is not implemented; use 'add' or None"
    )


def kthvalue(x, k, dim=-1, keepdim=False):
    """Return the ``k``-th smallest value (1-indexed) along ``dim``."""
    impl = _unwrap(x)
    ndim = len(impl.shape)
    d = int(dim) + ndim if int(dim) < 0 else int(dim)
    sorted_t = _C_engine.sort(impl, d)
    # ``sort`` returns just the sorted tensor; pick the (k-1)-th index.
    idx_shape = list(impl.shape)
    idx_shape[d] = 1
    idx_buf = _C_engine.full(idx_shape, float(int(k) - 1), _C_engine.I64, impl.device)
    val = _C_engine.gather(sorted_t, idx_buf, d)
    if not keepdim:
        val = _C_engine.squeeze(val, d)
    return _wrap(val)


# ── Comparison free functions (operators already work; these mirror the API) ──


def eq(x, y):
    """Elementwise equality. ``x`` and ``y`` may broadcast against each other."""
    return _wrap(_C_engine.equal(_unwrap(x), _unwrap(y) if hasattr(y, "_impl") else y))


def ne(x, y):
    """Elementwise inequality."""
    return _wrap(_C_engine.not_equal(_unwrap(x), _unwrap(y) if hasattr(y, "_impl") else y))


def lt(x, y):
    """Elementwise ``x < y``."""
    return _wrap(_C_engine.less(_unwrap(x), _unwrap(y) if hasattr(y, "_impl") else y))


def le(x, y):
    """Elementwise ``x <= y``."""
    return _wrap(_C_engine.less_equal(_unwrap(x), _unwrap(y) if hasattr(y, "_impl") else y))


def gt(x, y):
    """Elementwise ``x > y``."""
    return _wrap(_C_engine.greater(_unwrap(x), _unwrap(y) if hasattr(y, "_impl") else y))


def ge(x, y):
    """Elementwise ``x >= y``."""
    return _wrap(
        _C_engine.greater_equal(_unwrap(x), _unwrap(y) if hasattr(y, "_impl") else y)
    )


def isclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Elementwise ``|x − y| ≤ atol + rtol·|y|``."""
    import lucid as _l

    diff = _l.abs(x - y)
    tol = _l.abs(y) * float(rtol) + float(atol)
    out = diff <= tol
    if equal_nan:
        out = out | (_l.isnan(x) & _l.isnan(y))
    return out


# ── Logical / bitwise ─────────────────────────────────────────────────────────


def logical_and(x, y):
    """Boolean AND. Inputs are interpreted as ``x != 0``."""
    import lucid as _l

    bx = x != _l.zeros(1, dtype=x.dtype) if not hasattr(x, "dtype") else x != 0
    by = y != _l.zeros(1, dtype=y.dtype) if not hasattr(y, "dtype") else y != 0
    return bx & by


def logical_or(x, y):
    """Boolean OR."""
    bx = x != 0
    by = y != 0
    return bx | by


def logical_xor(x, y):
    """Boolean XOR."""
    bx = x != 0
    by = y != 0
    return bx ^ by


def logical_not(x):
    """Boolean NOT. Output is the bool dtype mask of ``x == 0``."""
    return x == 0


def bitwise_not(x):
    """Bitwise NOT (``~x``) for integer/bool tensors."""
    return ~x


def bitwise_and(x, y):
    """Elementwise bitwise AND."""
    return _wrap(_C_engine.bitwise_and(_unwrap(x), _unwrap(y) if hasattr(y, "_impl") else y))


def bitwise_or(x, y):
    """Elementwise bitwise OR."""
    return _wrap(_C_engine.bitwise_or(_unwrap(x), _unwrap(y) if hasattr(y, "_impl") else y))


def bitwise_xor(x, y):
    """Elementwise bitwise XOR."""
    return _wrap(_C_engine.bitwise_xor(_unwrap(x), _unwrap(y) if hasattr(y, "_impl") else y))


def cross(a, b, dim=-1):
    """Cross product alias delegating to ``lucid.linalg.cross`` so the
    standard tensor framework's top-level ``cross`` surface is preserved."""
    from lucid.linalg import cross as _cross

    return _cross(a, b, dim=dim)


def norm(x, ord=None, dim=None, keepdim=False):
    """Tensor norm — top-level alias for ``lucid.linalg.norm``."""
    from lucid.linalg import norm as _norm

    return _norm(x, ord=ord, dim=dim, keepdim=keepdim)


# ── Math (unary) extras ───────────────────────────────────────────────────────


def asin(x):
    """``asin`` alias for ``arcsin``."""
    return _wrap(_C_engine.arcsin(_unwrap(x)))


def acos(x):
    """``acos`` alias for ``arccos``."""
    return _wrap(_C_engine.arccos(_unwrap(x)))


def atan(x):
    """``atan`` alias for ``arctan``."""
    return _wrap(_C_engine.arctan(_unwrap(x)))


def log10(x):
    """Base-10 logarithm via ``log(x) / log(10)``."""
    import math
    import lucid as _l

    return _l.log(x) / float(math.log(10.0))


def log1p(x):
    """``log(1 + x)`` — the engine has no fused kernel so this composes."""
    import lucid as _l

    return _l.log(1.0 + x)


def exp2(x):
    """``2 ** x``."""
    import lucid as _l

    return _l.pow(2.0 + _l.zeros_like(x), x) if False else _wrap(
        _C_engine.exp(_C_engine.mul(_unwrap(x), _C_engine.full(list(_unwrap(x).shape), 0.6931471805599453, _unwrap(x).dtype, _unwrap(x).device)))
    )


def trunc(x):
    """Round each element toward zero."""
    import lucid as _l

    return _l.where(x >= 0, _l.floor(x), _l.ceil(x))


def frac(x):
    """Fractional part: ``x − trunc(x)``."""
    return x - trunc(x)


# ── Math (binary) extras ──────────────────────────────────────────────────────


def atan2(y, x):
    """Elementwise ``atan2(y, x)`` — quadrant-aware arctan via composition.

    Uses the standard formula with ``where`` branches; differentiable through
    the underlying ``arctan``, ``log``, and arithmetic ops.
    """
    import math
    import lucid as _l

    pi = float(math.pi)
    # Base value: arctan(y / x) — undefined when x == 0, handled below.
    safe_x = _l.where(x == 0, _l.ones_like(x), x)
    base = _l.arctan(y / safe_x)
    # Quadrant correction.
    res = _l.where(x > 0, base, base + pi)  # x>0 → base; x<0 → base+pi
    res = _l.where(
        (x < 0) & (y < 0), base - pi, res
    )  # x<0, y<0 → base − pi
    res = _l.where((x == 0) & (y > 0), _l.ones_like(x) * (pi / 2.0), res)
    res = _l.where((x == 0) & (y < 0), _l.ones_like(x) * (-pi / 2.0), res)
    res = _l.where((x == 0) & (y == 0), _l.zeros_like(x), res)
    return res


def fmod(x, y):
    """C-style modulo — result has the sign of ``x``."""
    import lucid as _l

    return x - trunc(x / y) * y


def remainder(x, y):
    """Python-style modulo — result has the sign of ``y``."""
    import lucid as _l

    return x - _l.floor(x / y) * y


def hypot(x, y):
    """``sqrt(x² + y²)`` (no overflow protection — same as the engine ops)."""
    import lucid as _l

    return _l.sqrt(x * x + y * y)


def logaddexp(x, y):
    """Numerically stable ``log(exp(x) + exp(y))``."""
    import lucid as _l

    m = _l.maximum(x, y)
    return m + _l.log(_l.exp(x - m) + _l.exp(y - m))


# ── Reductions extras ─────────────────────────────────────────────────────────


def logsumexp(x, dim=None, keepdim=False):
    """Numerically stable ``log(sum(exp(x), dim))``."""
    import lucid as _l

    axes = _to_axes(dim)
    m = _l.max(x, dim=axes if axes else None, keepdim=True)
    shifted = _l.exp(x - m)
    s = _l.sum(shifted, dim=axes if axes else None, keepdim=True)
    out = _l.log(s) + m
    if not keepdim:
        if axes:
            out = _l.squeeze(out, dim=axes)
        else:
            out = _l.squeeze(out)
    return out


# ── Linear-algebra extras ─────────────────────────────────────────────────────


def mm(a, b):
    """2-D matrix multiply. Both inputs must be rank-2."""
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError(
            f"mm: both inputs must be 2-D, got shapes {tuple(a.shape)} and {tuple(b.shape)}"
        )
    return _wrap(_C_engine.matmul(_unwrap(a), _unwrap(b)))


def bmm(a, b):
    """Batched matrix multiply. Both inputs must be rank-3 with matching batch."""
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise ValueError(
            f"bmm: both inputs must be 3-D, got shapes {tuple(a.shape)} and {tuple(b.shape)}"
        )
    return _wrap(_C_engine.matmul(_unwrap(a), _unwrap(b)))


def einsum(equation, *operands):
    """Einstein summation. Identical surface to ``lucid.einops.einsum``."""
    from lucid._C.engine import einops as _einops_engine

    return _wrap(_einops_engine.einsum(equation, [_unwrap(t) for t in operands]))


def kron(a, b):
    """Kronecker product of two tensors with matching ranks.

    For 2-D ``a`` (m, n) and ``b`` (p, q) the result has shape (m·p, n·q).
    """
    impl_a = _unwrap(a)
    impl_b = _unwrap(b)
    if len(impl_a.shape) != len(impl_b.shape):
        raise ValueError(
            f"kron: ranks must match, got {len(impl_a.shape)} and {len(impl_b.shape)}"
        )
    ndim = len(impl_a.shape)
    # Outer-product trick: reshape a → (..., 1) and b → (1, ...) along every axis,
    # multiply with broadcasting, then collapse the interleaved pairs.
    shape_a = list(impl_a.shape)
    shape_b = list(impl_b.shape)
    a_expanded_shape = [
        s for ax in range(ndim) for s in (shape_a[ax], 1)
    ]
    b_expanded_shape = [
        s for ax in range(ndim) for s in (1, shape_b[ax])
    ]
    a_r = _C_engine.reshape(impl_a, a_expanded_shape)
    b_r = _C_engine.reshape(impl_b, b_expanded_shape)
    prod = _C_engine.mul(a_r, b_r)
    out_shape = [shape_a[ax] * shape_b[ax] for ax in range(ndim)]
    return _wrap(_C_engine.reshape(prod, out_shape))


# ── Stats / search extras ─────────────────────────────────────────────────────


def searchsorted(sorted_sequence, values, *, right=False):
    """Return indices where ``values`` would be inserted into ``sorted_sequence``.

    Uses a numpy fallback because the engine has no native binary search; the
    call materialises both inputs to host memory.
    """
    import numpy as _np
    import lucid as _l

    seq = sorted_sequence.numpy() if hasattr(sorted_sequence, "numpy") else _np.asarray(sorted_sequence)
    vals = values.numpy() if hasattr(values, "numpy") else _np.asarray(values)
    side = "right" if right else "left"
    idx = _np.searchsorted(seq, vals, side=side).astype(_np.int64)
    return _l.tensor(idx)


def bucketize(x, boundaries, *, right=False):
    """Identical to ``searchsorted(boundaries, x, right=right)``."""
    return searchsorted(boundaries, x, right=right)


def histc(x, bins=100, min=0.0, max=0.0):
    """Compute a 1-D histogram of ``x`` with ``bins`` equal-width buckets.

    When ``min == max`` the range is taken from ``x.min()`` / ``x.max()`` —
    matches the standard tensor framework's ``histc`` default.
    """
    import lucid as _l

    if float(min) == float(max):
        lo = float(_l.min(x).item())
        hi = float(_l.max(x).item())
        if lo == hi:
            hi = lo + 1.0
    else:
        lo, hi = float(min), float(max)
    counts, _ = _C_engine.histogram(_unwrap(x), int(bins), lo, hi, False)
    return _wrap(counts)


def cartesian_prod(*tensors):
    """Cartesian product of 1-D tensors → 2-D tensor of every combination."""
    import lucid as _l

    if not tensors:
        raise ValueError("cartesian_prod requires at least one tensor")
    grids = _l.meshgrid(*tensors, indexing="ij")
    flat = [_l.reshape(g, [-1]) for g in grids]
    return stack_with_dim(flat, -1)


def stack_with_dim(tensors, dim):
    """Internal helper to call the engine's stack with a non-default axis."""
    impls = [_unwrap(t) for t in tensors]
    return _wrap(_C_engine.stack(impls, int(dim)))


_OVERRIDES = [
    "sum",
    "mean",
    "prod",
    "max",
    "min",
    "var",
    "std",
    "argmax",
    "argmin",
    "squeeze",
    "repeat",
    "repeat_interleave",
    "split",
    "tensordot",
    "meshgrid",
    "where",
    "masked_fill",
    "pad",
    "reshape",
    "permute",
    "expand",
    # ── new additions: shape/view aliases ─────────────────────────────────
    "view",
    "concat",
    "narrow",
    "movedim",
    "unflatten",
    # ── indexing extras ───────────────────────────────────────────────────
    "take",
    "index_select",
    "masked_select",
    "scatter",
    "kthvalue",
    # ── comparison free functions ─────────────────────────────────────────
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "isclose",
    # ── logical / bitwise ─────────────────────────────────────────────────
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "bitwise_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "cross",
    "norm",
    # ── math unary ────────────────────────────────────────────────────────
    "asin",
    "acos",
    "atan",
    "log10",
    "log1p",
    "exp2",
    "trunc",
    "frac",
    # ── math binary ───────────────────────────────────────────────────────
    "atan2",
    "fmod",
    "remainder",
    "hypot",
    "logaddexp",
    # ── reduction extras ──────────────────────────────────────────────────
    "logsumexp",
    # ── linear algebra ────────────────────────────────────────────────────
    "mm",
    "bmm",
    "einsum",
    "kron",
    # ── stats / search ────────────────────────────────────────────────────
    "searchsorted",
    "bucketize",
    "histc",
    "cartesian_prod",
]
for _name in _OVERRIDES:
    globals()[_name] = globals()[_name]  # already defined above
    _FREE_FN_NAMES.add(_name)

__all__ = sorted(_FREE_FN_NAMES)
