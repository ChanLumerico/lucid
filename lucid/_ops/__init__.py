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


# ── PyTorch-compatible overrides ──────────────────────────────────────────────
# These replace the auto-generated functions above with wrappers that accept
# PyTorch-style keyword arguments (dim, keepdim, correction, dims, etc.).


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
    """Sum; accepts PyTorch-style dim/keepdim."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _wrap(_C_engine.sum(_unwrap(x), ax, kd))


def mean(x, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """Mean; accepts PyTorch-style dim/keepdim."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _wrap(_C_engine.mean(_unwrap(x), ax, kd))


def prod(x, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """Product; accepts PyTorch-style dim/keepdim."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _wrap(_C_engine.prod(_unwrap(x), ax, kd))


def max(x, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """Max; accepts PyTorch-style dim/keepdim."""
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    return _wrap(_C_engine.max(_unwrap(x), ax, kd))


def min(x, dim=None, keepdim=False, *, axis=None, axes=None, keepdims=None):
    """Min; accepts PyTorch-style dim/keepdim."""
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
    """Variance; correction=1 applies Bessel's correction (PyTorch default)."""
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
    """Std dev; correction=1 applies Bessel's correction (PyTorch default)."""
    if unbiased is not None:
        correction = 1 if unbiased else 0
    ax = _to_axes(dim if dim is not None else axis if axis is not None else axes)
    kd = keepdims if keepdims is not None else keepdim
    impl = _unwrap(x)
    v = _C_engine.var(impl, ax, kd)
    v = _bessel_correct(v, impl, ax, correction)
    return _wrap(_C_engine.sqrt(v))


def argmax(x, dim=None, keepdim=False, *, axis=None, keepdims=None):
    """Argmax; accepts PyTorch-style dim/keepdim (axis also accepted)."""
    d = dim if dim is not None else axis
    kd = keepdims if keepdims is not None else keepdim
    if d is None:
        return _wrap(_C_engine.argmax(_unwrap(x), -1, kd))
    return _wrap(_C_engine.argmax(_unwrap(x), int(d), kd))


def argmin(x, dim=None, keepdim=False, *, axis=None, keepdims=None):
    """Argmin; accepts PyTorch-style dim/keepdim (axis also accepted)."""
    d = dim if dim is not None else axis
    kd = keepdims if keepdims is not None else keepdim
    if d is None:
        return _wrap(_C_engine.argmin(_unwrap(x), -1, kd))
    return _wrap(_C_engine.argmin(_unwrap(x), int(d), kd))


# ── shape ─────────────────────────────────────────────────────────────────────


def squeeze(x, dim=None):
    """Squeeze; dim=None removes all size-1 dims; list squeezes multiple dims.
    Non-unit dims are silently ignored (PyTorch behaviour).
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
    """Alias for repeat_interleave: repeats elements along dim (NumPy / PyTorch semantics).

    Note: matches torch.repeat_interleave, NOT Tensor.repeat (which tiles copies).
    Use lucid.tile(x, reps) to tile copies in each dimension.
    """
    impl = _unwrap(x)
    axis = 0 if dim is None else int(dim)
    return _wrap(_C_engine.repeat(impl, int(repeats), axis))


def repeat_interleave(x, repeats, dim=None):
    """Repeat elements of a tensor (interleave, matches torch.repeat_interleave)."""
    impl = _unwrap(x)
    axis = 0 if dim is None else int(dim)
    return _wrap(_C_engine.repeat(impl, int(repeats), axis))


# ── split ─────────────────────────────────────────────────────────────────────


def split(x, split_size_or_sections, dim=0):
    """Split a tensor into chunks (PyTorch semantics: split_size = chunk size).

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
    """Tensordot with PyTorch-compatible dims argument.

    Accepted call forms:
      tensordot(a, b, dims=2)               → contract last 2 of a with first 2 of b
      tensordot(a, b, dims=[[0,1],[2,3]])   → explicit axes lists (PyTorch style)
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
    Condition is cast to bool dtype if needed (mirrors PyTorch behaviour).
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


# ── pad — PyTorch flat format → engine per-dim pairs ─────────────────────────


def pad(x, padding, mode="constant", value=0.0):
    """Pad tensor using PyTorch flat convention (last-dim first).
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
]
for _name in _OVERRIDES:
    globals()[_name] = globals()[_name]  # already defined above
    _FREE_FN_NAMES.add(_name)

__all__ = sorted(_FREE_FN_NAMES)
