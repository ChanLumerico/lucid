"""
Tensor indexing: __getitem__ and __setitem__ implementation.

Supported index forms:
- int:       t[0]
- slice:     t[1:3], t[::2]
- tuple:     t[0, :, None]
- Ellipsis:  t[..., 0]
- None:      t[None]  (unsqueeze at dim 0)
- bool mask: t[mask]  (gather via nonzero)
- int Tensor: t[idx]  (advanced integer indexing)
"""

from typing import TYPE_CHECKING
import numpy as np
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap, _unwrap
from lucid._dtype import bool_ as _bool_dtype

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid._types import _IndexType, TensorOrScalar


def _select_int(impl: _C_engine.TensorImpl, dim: int, i: int) -> _C_engine.TensorImpl:
    """Select a single index along dim, removing that dimension."""
    length = impl.shape[dim]
    normalized = i + length if i < 0 else i
    if normalized < 0 or normalized >= length:
        raise IndexError(
            f"index {i} is out of bounds for dimension {dim} with size {length}"
        )
    # split_at to isolate [normalized : normalized+1], then squeeze
    hi = normalized + 1
    if normalized == 0 and hi == length:
        # only one element — squeeze the whole thing
        sliced = impl
    elif normalized == 0:
        sliced = _C_engine.split_at(impl, [hi], dim)[0]
    elif hi == length:
        sliced = _C_engine.split_at(impl, [normalized], dim)[1]
    else:
        sliced = _C_engine.split_at(impl, [normalized, hi], dim)[1]
    return _C_engine.squeeze(sliced, dim)


def _select_slice(
    impl: _C_engine.TensorImpl, dim: int, s: slice
) -> _C_engine.TensorImpl:
    """Slice along dim using a Python slice object."""
    length = impl.shape[dim]
    start, stop, step = s.indices(length)

    if step == 1:
        # Fast path via split_at
        if start == 0 and stop == length:
            return impl  # no-op
        if start == 0:
            return _C_engine.split_at(impl, [stop], dim)[0]
        if stop == length:
            return _C_engine.split_at(impl, [start], dim)[1]
        return _C_engine.split_at(impl, [start, stop], dim)[1]
    else:
        # Strided: build index array and gather with broadcast
        indices_1d = np.arange(start, stop, step, dtype=np.int32)
        n = len(indices_1d)
        if n == 0:
            shape = list(impl.shape)
            shape[dim] = 0
            empty_arr = np.zeros(shape, dtype=np.float32)
            return _C_engine.TensorImpl(empty_arr, impl.device, False)
        # Broadcast 1D indices to the full output shape
        out_shape = list(impl.shape)
        out_shape[dim] = n
        bcast_shape = [1] * len(out_shape)
        bcast_shape[dim] = n
        idx_nd = np.broadcast_to(indices_1d.reshape(bcast_shape), out_shape).copy()
        idx_impl = _C_engine.TensorImpl(
            np.ascontiguousarray(idx_nd), impl.device, False
        )
        return _C_engine.gather(impl, idx_impl, dim)


def _getitem(t: Tensor, idx: _IndexType) -> Tensor:
    impl = t._impl

    # bool mask
    if hasattr(idx, "_impl") and getattr(idx, "dtype", None) is not None:
        if idx.dtype is _bool_dtype:
            indices = _C_engine.nonzero(_unwrap(idx))
            return _wrap(_C_engine.gather(impl, indices, 0))
        # integer tensor indexing
        return _wrap(_C_engine.gather(impl, _unwrap(idx), 0))

    # normalize single index → tuple
    if not isinstance(idx, tuple):
        idx = (idx,)

    result_impl = _normalize_and_apply_index(impl, idx)
    return _wrap(result_impl)


def _normalize_and_apply_index(
    impl: _C_engine.TensorImpl,
    idx_tuple: tuple[object, ...],
) -> _C_engine.TensorImpl:
    """Apply a normalized index tuple to a TensorImpl."""
    ndim = len(impl.shape)

    n_none = sum(1 for i in idx_tuple if i is None)
    n_ellipsis = sum(1 for i in idx_tuple if i is ...)
    n_real = len(idx_tuple) - n_none - n_ellipsis
    ellipsis_len = max(ndim - n_real, 0)

    # expand Ellipsis
    expanded: list[int | slice | Tensor | list[int] | None] = []
    for i in idx_tuple:
        if i is ...:
            expanded.extend([slice(None)] * ellipsis_len)
        else:
            expanded.append(i)

    dim = 0
    for i in expanded:
        if i is None:
            impl = _C_engine.unsqueeze(impl, dim)
            dim += 1
        elif isinstance(i, int):
            impl = _select_int(impl, dim, i)
            # dim does NOT advance — select removed this dimension
        elif isinstance(i, slice):
            impl = _select_slice(impl, dim, i)
            dim += 1
        else:
            raise IndexError(f"Unsupported index type: {type(i).__name__}")

    return impl


def _setitem(t: Tensor, idx: _IndexType, value: TensorOrScalar) -> None:
    """In-place assignment: t[idx] = value."""
    if hasattr(value, "_impl"):
        v_impl = value._impl
    elif isinstance(value, (int, float, bool)):
        v_impl = _C_engine.full(
            t._impl.shape, float(value), t._impl.dtype, t._impl.device
        )
    else:
        raise TypeError(f"Cannot assign value of type {type(value).__name__}")

    if hasattr(_C_engine, "index_put_"):
        _C_engine.index_put_(t._impl, idx, v_impl)
    else:
        raise NotImplementedError(
            "Tensor __setitem__ requires engine.index_put_ support"
        )
