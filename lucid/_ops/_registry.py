"""
Ops registry: maps op names to engine functions and Tensor method names.

Used by _methods.py to auto-inject Tensor methods and by _ops/__init__.py
to expose free functions.
"""

from dataclasses import dataclass, field
from typing import Callable, Any

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap  # needed for mixed-arg adapters

# ── Adapters for ops whose arg order differs from the engine signature ────────


def _detach_adapter(impl):
    """detach(x): deep-copy without gradient tracking."""
    return _C_engine.contiguous(impl).clone_with_grad(False)


def _scatter_add_adapter(x_impl, dim: int, index, src):
    """scatter_add(x, dim, index, src) — Python order → engine order."""
    # Engine: scatter_add(base, indices, src, dim)
    return _C_engine.scatter_add(x_impl, _unwrap(index), _unwrap(src), dim)


# ── Adapters for composite ops whose Python signature differs from engine ─────


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


def _logsumexp_adapter(a_impl, dim=None, keepdim=False):
    """logsumexp(a, dim=None, keepdim=False) — accept axis/None like sum/mean."""
    if dim is None:
        axes = []
    elif isinstance(dim, (list, tuple)):
        axes = [int(d) for d in dim]
    else:
        axes = [int(dim)]
    return _C_engine.logsumexp(a_impl, axes, bool(keepdim))


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


# ── Adapters for top-level aliases of linalg sub-module functions ────────────


def _cross_adapter(a_impl, b_impl, dim: int = -1):
    """cross(a, b, dim=-1) — forwards to lucid.linalg.cross.

    Lives in the registry because ``lucid.linalg.cross`` is a Python wrapper
    around an engine kernel, and we want the top-level ``lucid.cross`` to
    behave identically.
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


# ── Reference-framework-compatible reductions / shape adapters ───────────────
#
# These ops take ``dim``/``axis``/``keepdim``/``correction`` keyword arguments
# at the public surface but the engine kernels accept positional list[int]
# axes plus a bool keepdim.  The adapters below normalise the signatures so
# the registry stays a single source of truth — no parallel definitions in
# ``_ops/__init__.py``.


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


def _sum_adapter(x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """sum(x, dim=None, keepdim=False) with axis/axes/keepdims aliases."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.sum(x_impl, ax, kd)


def _mean_adapter(x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """mean(x, dim=None, keepdim=False) with axis/axes/keepdims aliases."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.mean(x_impl, ax, kd)


def _prod_adapter(x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """prod(x, dim=None, keepdim=False)."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.prod(x_impl, ax, kd)


def _max_adapter(x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """max(x, dim=None, keepdim=False)."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.max(x_impl, ax, kd)


def _min_adapter(x_impl, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """min(x, dim=None, keepdim=False)."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _C_engine.min(x_impl, ax, kd)


def _var_adapter(x_impl, dim=None, keepdim=False, *,
                 correction=1, unbiased=None, axis=None, axes=None, keepdims=None):
    """var(x, dim, keepdim, correction=1) — ddof default matches reference."""
    if unbiased is not None:
        correction = 1 if unbiased else 0
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    result = _C_engine.var(x_impl, ax, kd)
    return _bessel_correct(result, x_impl, ax, correction)


def _std_adapter(x_impl, dim=None, keepdim=False, *,
                 correction=1, unbiased=None, axis=None, axes=None, keepdims=None):
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


def _repeat_adapter(x_impl, repeats, dim=None):
    """repeat(x, repeats, dim=None) — interleaved replication along axis 0
    when ``dim`` is None, else along the given axis."""
    axis = 0 if dim is None else int(dim)
    return _C_engine.repeat(x_impl, int(repeats), axis)


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


@dataclass
class OpEntry:
    """Descriptor for a single operation."""

    name: str
    engine_fn: Callable[..., Any]
    n_tensor_args: int  # positional Tensor args to unwrap; -1 = first arg is list
    returns_tensor: bool = True
    inplace: bool = False
    method_name: str | None = None  # Tensor method name (None = no method)
    free_fn_name: str | None = None  # lucid.xxx name (None = same as name)
    extra_kwargs: list[str] = field(default_factory=list)


_R = _C_engine  # shorthand

# fmt: off
_REGISTRY: list[OpEntry] = [
    # ── unary ──────────────────────────────────────────────────────────────
    OpEntry("neg",       _R.neg,       1, method_name="neg",       free_fn_name="neg"),
    OpEntry("abs",       _R.abs,       1, method_name="abs",       free_fn_name="abs"),
    OpEntry("sign",      _R.sign,      1, method_name="sign",      free_fn_name="sign"),
    OpEntry("exp",       _R.exp,       1, method_name="exp",       free_fn_name="exp"),
    OpEntry("log",       _R.log,       1, method_name="log",       free_fn_name="log"),
    OpEntry("log2",      _R.log2,      1, method_name="log2",      free_fn_name="log2"),
    OpEntry("sqrt",      _R.sqrt,      1, method_name="sqrt",      free_fn_name="sqrt"),
    # rsqrt: not in engine — implemented in Python as reciprocal(sqrt(x))
    OpEntry("square",    _R.square,    1, method_name="square",    free_fn_name="square"),
    OpEntry("reciprocal",_R.reciprocal,1, method_name="reciprocal",free_fn_name="reciprocal"),
    OpEntry("floor",     _R.floor,     1, method_name="floor",     free_fn_name="floor"),
    OpEntry("ceil",      _R.ceil,      1, method_name="ceil",      free_fn_name="ceil"),
    OpEntry("round",     _R.round,     1, method_name="round",     free_fn_name="round"),
    OpEntry("sin",       _R.sin,       1, method_name="sin",       free_fn_name="sin"),
    OpEntry("cos",       _R.cos,       1, method_name="cos",       free_fn_name="cos"),
    OpEntry("tan",       _R.tan,       1, method_name="tan",       free_fn_name="tan"),
    OpEntry("arcsin",    _R.arcsin,    1, method_name="arcsin",    free_fn_name="arcsin"),
    OpEntry("arccos",    _R.arccos,    1, method_name="arccos",    free_fn_name="arccos"),
    OpEntry("arctan",    _R.arctan,    1, method_name="arctan",    free_fn_name="arctan"),
    OpEntry("sinh",      _R.sinh,      1, method_name="sinh",      free_fn_name="sinh"),
    OpEntry("cosh",      _R.cosh,      1, method_name="cosh",      free_fn_name="cosh"),
    OpEntry("tanh",      _R.tanh,      1, method_name="tanh",      free_fn_name="tanh"),
    OpEntry("relu",      _R.relu,      1, method_name="relu",      free_fn_name="relu"),
    OpEntry("sigmoid",   _R.sigmoid,   1, method_name="sigmoid",   free_fn_name="sigmoid"),
    OpEntry("silu",      _R.silu,      1, method_name="silu",      free_fn_name="silu"),
    OpEntry("gelu",      _R.gelu,      1, method_name="gelu",      free_fn_name="gelu"),
    OpEntry("mish",      _R.mish,      1, method_name="mish",      free_fn_name="mish"),
    OpEntry("selu",      _R.selu,      1, method_name="selu",      free_fn_name="selu"),
    OpEntry("softplus",  _R.softplus,  1, method_name="softplus",  free_fn_name="softplus"),
    OpEntry("relu6",     _R.relu6,     1, method_name="relu6",     free_fn_name="relu6"),
    OpEntry("hard_sigmoid", _R.hard_sigmoid, 1, method_name="hard_sigmoid", free_fn_name="hard_sigmoid"),
    OpEntry("hard_swish",   _R.hard_swish,   1, method_name="hard_swish",   free_fn_name="hard_swish"),
    OpEntry("ravel",     _R.ravel,     1, method_name="ravel",     free_fn_name="ravel"),
    OpEntry("contiguous",_R.contiguous,1, method_name="contiguous",free_fn_name="contiguous"),

    # ── in-place unary ─────────────────────────────────────────────────────
    OpEntry("neg_",        _R.neg_,        1, inplace=True, method_name="neg_"),
    OpEntry("abs_",        _R.abs_,        1, inplace=True, method_name="abs_"),
    OpEntry("sign_",       _R.sign_,       1, inplace=True, method_name="sign_"),
    OpEntry("reciprocal_", _R.reciprocal_, 1, inplace=True, method_name="reciprocal_"),
    OpEntry("square_",     _R.square_,     1, inplace=True, method_name="square_"),
    OpEntry("cube_",       _R.cube_,       1, inplace=True, method_name="cube_"),
    OpEntry("exp_",        _R.exp_,        1, inplace=True, method_name="exp_"),
    OpEntry("log_",        _R.log_,        1, inplace=True, method_name="log_"),
    OpEntry("log2_",       _R.log2_,       1, inplace=True, method_name="log2_"),
    OpEntry("sqrt_",       _R.sqrt_,       1, inplace=True, method_name="sqrt_"),
    OpEntry("sin_",        _R.sin_,        1, inplace=True, method_name="sin_"),
    OpEntry("cos_",        _R.cos_,        1, inplace=True, method_name="cos_"),
    OpEntry("tan_",        _R.tan_,        1, inplace=True, method_name="tan_"),
    OpEntry("arcsin_",     _R.arcsin_,     1, inplace=True, method_name="arcsin_"),
    OpEntry("arccos_",     _R.arccos_,     1, inplace=True, method_name="arccos_"),
    OpEntry("arctan_",     _R.arctan_,     1, inplace=True, method_name="arctan_"),
    OpEntry("sinh_",       _R.sinh_,       1, inplace=True, method_name="sinh_"),
    OpEntry("cosh_",       _R.cosh_,       1, inplace=True, method_name="cosh_"),
    OpEntry("tanh_",       _R.tanh_,       1, inplace=True, method_name="tanh_"),
    OpEntry("sigmoid_",    _R.sigmoid_,    1, inplace=True, method_name="sigmoid_"),
    OpEntry("relu_",       _R.relu_,       1, inplace=True, method_name="relu_"),
    OpEntry("floor_",      _R.floor_,      1, inplace=True, method_name="floor_"),
    OpEntry("ceil_",       _R.ceil_,       1, inplace=True, method_name="ceil_"),
    OpEntry("round_",      _R.round_,      1, inplace=True, method_name="round_"),

    # ── binary ─────────────────────────────────────────────────────────────
    OpEntry("add",      _R.add,      2, method_name="add",      free_fn_name="add"),
    OpEntry("sub",      _R.sub,      2, method_name="sub",      free_fn_name="sub"),
    OpEntry("mul",      _R.mul,      2, method_name="mul",      free_fn_name="mul"),
    OpEntry("div",      _R.div,      2, method_name="div",      free_fn_name="div"),
    OpEntry("pow",      _R.pow,      2, method_name="pow",      free_fn_name="pow"),
    OpEntry("maximum",  _R.maximum,  2, method_name="maximum",  free_fn_name="maximum"),
    OpEntry("minimum",  _R.minimum,  2, method_name="minimum",  free_fn_name="minimum"),
    OpEntry("matmul",   _R.matmul,   2, method_name="matmul",   free_fn_name="matmul"),
    OpEntry("dot",      _R.dot,      2, method_name="dot",      free_fn_name="dot"),
    OpEntry("inner",    _R.inner,    2, method_name="inner",    free_fn_name="inner"),
    OpEntry("outer",    _R.outer,    2, method_name="outer",    free_fn_name="outer"),

    # ── in-place binary ────────────────────────────────────────────────────
    OpEntry("add_",     _R.add_,     2, inplace=True, method_name="add_"),
    OpEntry("sub_",     _R.sub_,     2, inplace=True, method_name="sub_"),
    OpEntry("mul_",     _R.mul_,     2, inplace=True, method_name="mul_"),
    OpEntry("div_",     _R.div_,     2, inplace=True, method_name="div_"),
    OpEntry("pow_",     _R.pow_,     2, inplace=True, method_name="pow_"),
    OpEntry("maximum_", _R.maximum_, 2, inplace=True, method_name="maximum_"),
    OpEntry("minimum_", _R.minimum_, 2, inplace=True, method_name="minimum_"),

    # ── reduction (with API-compat adapters) ───────────────────────────────
    OpEntry("sum",    _sum_adapter,    1, method_name="sum",    free_fn_name="sum",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("mean",   _mean_adapter,   1, method_name="mean",   free_fn_name="mean",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("prod",   _prod_adapter,   1, method_name="prod",   free_fn_name="prod",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("max",    _max_adapter,    1, method_name="max",    free_fn_name="max",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("min",    _min_adapter,    1, method_name="min",    free_fn_name="min",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("var",    _var_adapter,    1, method_name="var",    free_fn_name="var",
            extra_kwargs=["dim", "keepdim", "correction", "unbiased",
                          "axis", "axes", "keepdims"]),
    OpEntry("argmax", _argmax_adapter, 1, method_name="argmax", free_fn_name="argmax",
            extra_kwargs=["dim", "keepdim", "axis", "keepdims"]),
    OpEntry("argmin", _argmin_adapter, 1, method_name="argmin", free_fn_name="argmin",
            extra_kwargs=["dim", "keepdim", "axis", "keepdims"]),
    OpEntry("cumsum", _R.cumsum, 1, method_name="cumsum", free_fn_name="cumsum",
            extra_kwargs=["axis"]),
    OpEntry("cumprod",_R.cumprod,1, method_name="cumprod",free_fn_name="cumprod",
            extra_kwargs=["axis"]),
    OpEntry("trace",  _R.trace,  1, method_name="trace",  free_fn_name="trace"),

    # ── shape / layout ─────────────────────────────────────────────────────
    OpEntry("reshape",    _reshape_adapter, 1, method_name="reshape",    free_fn_name="reshape"),
    OpEntry("squeeze",    _squeeze_adapter, 1, method_name="squeeze",    free_fn_name="squeeze",
            extra_kwargs=["dim"]),
    OpEntry("squeeze_all",_R.squeeze_all,1, method_name="squeeze_all"),
    OpEntry("unsqueeze",  _R.unsqueeze,  1, method_name="unsqueeze",  free_fn_name="unsqueeze",
            extra_kwargs=["dim"]),
    OpEntry("flatten",    _R.flatten,    1, method_name="flatten",    free_fn_name="flatten",
            extra_kwargs=["start", "end"]),
    OpEntry("permute",    _permute_adapter, 1, method_name="permute",    free_fn_name="permute"),
    OpEntry("transpose",  _R.transpose,  1, method_name="transpose",  free_fn_name="transpose"),
    OpEntry("swapaxes",   _R.swapaxes,   1, method_name="swapaxes",
            extra_kwargs=["d0", "d1"]),  # positional: swapaxes(d0, d1)
    OpEntry("broadcast_to",_R.broadcast_to,1,method_name="broadcast_to",free_fn_name="broadcast_to",
            extra_kwargs=["shape"]),
    OpEntry("expand",     _expand_adapter, 1, method_name="expand",     free_fn_name="expand"),
    OpEntry("expand_dims",_R.expand_dims,1, method_name="expand_dims",
            extra_kwargs=["axis"]),
    OpEntry("repeat",     _repeat_adapter, 1, method_name="repeat",     free_fn_name="repeat",
            extra_kwargs=["dim"]),
    OpEntry("tile",       _R.tile,       1, method_name="tile",       free_fn_name="tile",
            extra_kwargs=["reps"]),
    OpEntry("roll",       _R.roll,       1, method_name="roll",       free_fn_name="roll",
            extra_kwargs=["shifts", "dims"]),
    OpEntry("tril",       _R.tril,       1, method_name="tril",       free_fn_name="tril",
            extra_kwargs=["k"]),
    OpEntry("triu",       _R.triu,       1, method_name="triu",       free_fn_name="triu",
            extra_kwargs=["k"]),
    OpEntry("pad",        _pad_adapter, 1, method_name="pad",        free_fn_name="pad",
            extra_kwargs=["padding", "mode", "value"]),

    # ── index / gather ─────────────────────────────────────────────────────
    OpEntry("gather",    _R.gather,    2, method_name="gather",    free_fn_name="gather",
            extra_kwargs=["dim"]),
    OpEntry("sort",      _R.sort,      1, method_name="sort",      free_fn_name="sort",
            extra_kwargs=["dim", "descending"]),
    OpEntry("argsort",   _R.argsort,   1, method_name="argsort",   free_fn_name="argsort",
            extra_kwargs=["dim", "descending"]),
    OpEntry("nonzero",   _R.nonzero,   1, method_name="nonzero",   free_fn_name="nonzero"),
    OpEntry("unique",    _R.unique,    1, method_name=None,        free_fn_name="unique"),
    OpEntry("topk",      _R.topk,      1, method_name="topk",      free_fn_name="topk",
            extra_kwargs=["k", "dim", "largest"]),
    OpEntry("diagonal",  _R.diagonal,  1, method_name="diagonal",  free_fn_name="diagonal",
            extra_kwargs=["offset", "dim1", "dim2"]),

    # ── comparison ────────────────────────────────────────────────────────
    OpEntry("equal",         _R.equal,         2, method_name=None, free_fn_name="equal"),
    OpEntry("not_equal",     _R.not_equal,     2, method_name=None, free_fn_name="not_equal"),
    OpEntry("greater",       _R.greater,       2, method_name=None, free_fn_name="greater"),
    OpEntry("greater_equal", _R.greater_equal, 2, method_name=None, free_fn_name="greater_equal"),
    OpEntry("less",          _R.less,          2, method_name=None, free_fn_name="less"),
    OpEntry("less_equal",    _R.less_equal,    2, method_name=None, free_fn_name="less_equal"),

    # ── masking ────────────────────────────────────────────────────────────
    # ``where`` and ``masked_fill`` auto-cast their condition/mask to bool to
    # match reference behaviour, so they need adapters rather than direct
    # engine bindings.
    OpEntry("where",       _where_adapter, 0, free_fn_name="where"),
    OpEntry("masked_fill", _masked_fill_adapter, 1,
            method_name="masked_fill", free_fn_name="masked_fill",
            extra_kwargs=["value"]),

    # ── joining ────────────────────────────────────────────────────────────
    OpEntry("concatenate", _R.concatenate, -1, free_fn_name="cat",
            extra_kwargs=["axis"]),
    OpEntry("stack",       _R.stack,       -1, free_fn_name="stack",
            extra_kwargs=["axis"]),
    OpEntry("hstack",      _R.hstack,      -1, free_fn_name="hstack"),
    OpEntry("vstack",      _R.vstack,      -1, free_fn_name="vstack"),
    OpEntry("split",       _split_adapter, 1,  method_name="split",      free_fn_name="split",
            extra_kwargs=["dim"]),
    OpEntry("chunk",       _R.chunk,       1,  method_name="chunk",      free_fn_name="chunk",
            extra_kwargs=["n", "axis"]),
    OpEntry("unbind",      _R.unbind,      1,  method_name="unbind",     free_fn_name="unbind",
            extra_kwargs=["axis"]),
    OpEntry("meshgrid",    _meshgrid_adapter, 0, free_fn_name="meshgrid",
            extra_kwargs=["indexing"]),

    # ── softmax / log_softmax (have axis kwarg) ────────────────────────────
    OpEntry("softmax",     _R.softmax,     1, method_name="softmax",     free_fn_name="softmax",
            extra_kwargs=["axis"]),
    OpEntry("log_softmax", _R.log_softmax, 1, method_name="log_softmax", free_fn_name="log_softmax",
            extra_kwargs=["axis"]),

    # ── rsqrt / std (engine-native) ────────────────────────────────────────
    OpEntry("rsqrt", _R.rsqrt, 1, method_name="rsqrt", free_fn_name="rsqrt"),
    OpEntry("std",   _std_adapter, 1, method_name="std",   free_fn_name="std",
            extra_kwargs=["dim", "keepdim", "correction", "unbiased",
                          "axis", "axes", "keepdims"]),

    # ── boolean reductions ─────────────────────────────────────────────────
    OpEntry("any", _R.any, 1, method_name="any", free_fn_name="any"),
    OpEntry("all", _R.all, 1, method_name="all", free_fn_name="all"),

    # ── linear algebra ─────────────────────────────────────────────────────
    OpEntry("tensordot",  _tensordot_adapter, 2, free_fn_name="tensordot",
            extra_kwargs=["dims"]),
    OpEntry("clip",       _R.clip,      1, method_name="clip",       free_fn_name="clip",
            extra_kwargs=["min", "max"]),

    # ── floating-point predicates (output is always bool) ──────────────────
    OpEntry("isinf",      _R.isinf,     1, method_name="isinf",     free_fn_name="isinf"),
    OpEntry("isnan",      _R.isnan,     1, method_name="isnan",     free_fn_name="isnan"),
    OpEntry("isfinite",   _R.isfinite,  1, method_name="isfinite",  free_fn_name="isfinite"),
    OpEntry("nan_to_num", _R.nan_to_num,1, method_name="nan_to_num",free_fn_name="nan_to_num",
            extra_kwargs=["nan", "posinf", "neginf"]),

    # ── tensor lifecycle ────────────────────────────────────────────────────
    # detach: deep-copy without gradient tracking (uses contiguous + clone_with_grad).
    OpEntry("detach",      _detach_adapter, 1, method_name="detach", free_fn_name="detach"),
    # clone: deep-copy preserving autograd history (contiguous = storage copy).
    OpEntry("clone",       _R.contiguous,   1, method_name="clone",  free_fn_name="clone"),
    # clamp is an alias for clip (same signature, same engine op).
    OpEntry("clamp",       _R.clip,         1, method_name="clamp",  free_fn_name="clamp",
            extra_kwargs=["min", "max"]),
    # scatter_add: arg order differs — adapter reorders before calling engine.
    # Python:  scatter_add(x, dim, index, src)
    # Engine:  scatter_add(base, indices, src, dim)
    # n_tensor_args=1 auto-unwraps x; the adapter manually unwraps index/src.
    OpEntry("scatter_add", _scatter_add_adapter, 1,
            method_name="scatter_add", free_fn_name="scatter_add"),

    # ══ composite ops (impl in _C/ops/composite/) ═══════════════════════════
    # Composition wrappers built atop primitives.  Autograd flows through the
    # underlying primitive backward nodes; no new schemas are registered.

    # ── elementwise math compositions (unary) ───────────────────────────────
    OpEntry("log10", _R.log10, 1, method_name="log10", free_fn_name="log10"),
    OpEntry("log1p", _R.log1p, 1, method_name="log1p", free_fn_name="log1p"),
    OpEntry("exp2",  _R.exp2,  1, method_name="exp2",  free_fn_name="exp2"),
    OpEntry("trunc", _R.trunc, 1, method_name="trunc", free_fn_name="trunc"),
    OpEntry("frac",  _R.frac,  1, method_name="frac",  free_fn_name="frac"),

    # ── elementwise math compositions (binary) ──────────────────────────────
    OpEntry("atan2",     _R.atan2,     2, method_name="atan2",     free_fn_name="atan2"),
    OpEntry("fmod",      _R.fmod,      2, method_name="fmod",      free_fn_name="fmod"),
    OpEntry("remainder", _R.remainder, 2, method_name="remainder", free_fn_name="remainder"),
    OpEntry("hypot",     _R.hypot,     2, method_name="hypot",     free_fn_name="hypot"),
    OpEntry("logaddexp", _R.logaddexp, 2, method_name="logaddexp", free_fn_name="logaddexp"),

    # ── reduction compositions ──────────────────────────────────────────────
    OpEntry("logsumexp", _logsumexp_adapter, 1,
            method_name="logsumexp", free_fn_name="logsumexp"),

    # ── linear-algebra compositions ─────────────────────────────────────────
    OpEntry("mm",   _R.mm,   2, method_name="mm",   free_fn_name="mm"),
    OpEntry("bmm",  _R.bmm,  2, method_name="bmm",  free_fn_name="bmm"),
    OpEntry("kron", _R.kron, 2, method_name="kron", free_fn_name="kron"),

    # ── logical compositions ────────────────────────────────────────────────
    OpEntry("logical_and", _R.logical_and, 2,
            method_name="logical_and", free_fn_name="logical_and"),
    OpEntry("logical_or",  _R.logical_or, 2,
            method_name="logical_or",  free_fn_name="logical_or"),
    OpEntry("logical_xor", _R.logical_xor, 2,
            method_name="logical_xor", free_fn_name="logical_xor"),
    OpEntry("logical_not", _R.logical_not, 1,
            method_name="logical_not", free_fn_name="logical_not"),

    # ── indexing compositions ───────────────────────────────────────────────
    OpEntry("take",         _take_adapter, 1,
            method_name="take", free_fn_name="take"),
    OpEntry("index_select", _index_select_adapter, 1,
            method_name="index_select", free_fn_name="index_select"),
    OpEntry("narrow",       _narrow_adapter, 1,
            method_name="narrow", free_fn_name="narrow"),
    OpEntry("scatter",      _scatter_adapter, 1,
            method_name="scatter", free_fn_name="scatter",
            extra_kwargs=["reduce"]),
    OpEntry("kthvalue",     _kthvalue_adapter, 1,
            method_name="kthvalue", free_fn_name="kthvalue"),

    # ── layout compositions ─────────────────────────────────────────────────
    OpEntry("movedim",   _movedim_adapter, 1,
            method_name="movedim", free_fn_name="movedim"),
    OpEntry("unflatten", _unflatten_adapter, 1,
            method_name="unflatten", free_fn_name="unflatten"),

    # ── stats / search compositions ─────────────────────────────────────────
    OpEntry("histc",         _histc_adapter, 1,
            method_name="histc", free_fn_name="histc"),
    OpEntry("cartesian_prod", _cartesian_prod_adapter, 0,
            method_name=None, free_fn_name="cartesian_prod"),
    OpEntry("searchsorted",  _searchsorted_adapter, 2,
            method_name=None, free_fn_name="searchsorted",
            extra_kwargs=["right"]),
    OpEntry("bucketize",     _bucketize_adapter, 2,
            method_name=None, free_fn_name="bucketize",
            extra_kwargs=["right"]),

    # ══ aliases of existing primitives (no new C++ — share engine kernels) ══

    # Comparison short names — same kernels as equal/not_equal/less/...
    OpEntry("eq", _R.eq, 2, method_name="eq", free_fn_name="eq"),
    OpEntry("ne", _R.ne, 2, method_name="ne", free_fn_name="ne"),
    OpEntry("lt", _R.lt, 2, method_name="lt", free_fn_name="lt"),
    OpEntry("le", _R.le, 2, method_name="le", free_fn_name="le"),
    OpEntry("gt", _R.gt, 2, method_name="gt", free_fn_name="gt"),
    OpEntry("ge", _R.ge, 2, method_name="ge", free_fn_name="ge"),

    # Trig short names — share kernels with arcsin/arccos/arctan.
    OpEntry("asin", _R.asin, 1, method_name="asin", free_fn_name="asin"),
    OpEntry("acos", _R.acos, 1, method_name="acos", free_fn_name="acos"),
    OpEntry("atan", _R.atan, 1, method_name="atan", free_fn_name="atan"),

    # ``bitwise_not`` is the reference framework's name for ``invert``.
    OpEntry("bitwise_not", _R.bitwise_not, 1,
            method_name="bitwise_not", free_fn_name="bitwise_not"),
    OpEntry("bitwise_and", _R.bitwise_and, 2,
            method_name="bitwise_and", free_fn_name="bitwise_and"),
    OpEntry("bitwise_or",  _R.bitwise_or,  2,
            method_name="bitwise_or",  free_fn_name="bitwise_or"),
    OpEntry("bitwise_xor", _R.bitwise_xor, 2,
            method_name="bitwise_xor", free_fn_name="bitwise_xor"),

    # masked_select: engine kernel returns a flat 1-D tensor of selected
    # elements.  Both inputs are tensors so n_tensor_args=2.
    OpEntry("masked_select", _R.masked_select, 2,
            method_name="masked_select", free_fn_name="masked_select"),

    # isclose: composite op with rtol/atol/equal_nan keyword arguments.
    # n_tensor_args=2 so both operands get unwrapped before the adapter runs.
    OpEntry("isclose", _isclose_adapter, 2,
            method_name="isclose", free_fn_name="isclose",
            extra_kwargs=["rtol", "atol", "equal_nan"]),

    # repeat_interleave: thin wrapper around engine.repeat that supports
    # ``dim=None`` (flatten first) like the reference framework's API.
    OpEntry("repeat_interleave", _repeat_interleave_adapter, 1,
            method_name="repeat_interleave", free_fn_name="repeat_interleave"),

    # Shape aliases.
    OpEntry("view",   _view_adapter,   1,
            method_name="view", free_fn_name="view"),
    OpEntry("concat", _concat_adapter, -1,
            method_name=None, free_fn_name="concat"),

    # ── top-level forwarders into the linalg sub-module ────────────────────
    OpEntry("cross", _cross_adapter, 2,
            method_name="cross", free_fn_name="cross"),
    OpEntry("norm",  _norm_adapter, 1,
            method_name="norm", free_fn_name="norm"),

    # ── top-level forwarder into the einops sub-module ─────────────────────
    # The user explicitly wanted ``lucid.einops.einsum`` to remain the
    # primary entry point; this top-level alias just gives Python users the
    # familiar ``lucid.einsum(...)`` shorthand without going through that
    # sub-module.  Both expose the same engine kernel.
    OpEntry("einsum", _einsum_adapter, 0,
            method_name=None, free_fn_name="einsum"),
]
# fmt: on
