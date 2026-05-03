"""
Tensor indexing: __getitem__ and __setitem__ implementation.

Supported index forms:
- int:       t[0]
- slice:     t[1:3]
- tuple:     t[0, :, None]
- Ellipsis:  t[..., 0]
- None:      t[None]  (unsqueeze at dim 0)
- bool mask: t[mask]  (1D gather via nonzero)
- int Tensor: t[idx]  (advanced integer indexing)
"""

from typing import Any, TYPE_CHECKING
from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _getitem(t: Tensor, idx: Any) -> Tensor:
    from lucid._dispatch import _wrap, _unwrap
    from lucid._tensor.tensor import Tensor as _Tensor

    impl = t._impl

    # bool mask
    if isinstance(idx, _Tensor) and idx.dtype._name == "bool":
        indices = _C_engine.nonzero(_unwrap(idx))
        return _wrap(_C_engine.gather(impl, 0, indices))

    # integer tensor indexing
    if isinstance(idx, _Tensor):
        return _wrap(_C_engine.gather(impl, 0, _unwrap(idx)))

    # normalize single index → tuple
    if not isinstance(idx, tuple):
        idx = (idx,)

    result_impl = _normalize_and_apply_index(impl, idx)
    return _wrap(result_impl)


def _normalize_and_apply_index(
    impl: _C_engine.TensorImpl,
    idx_tuple: tuple[Any, ...],
) -> _C_engine.TensorImpl:
    """Apply a normalized index tuple to a TensorImpl."""
    ndim = len(impl.shape)

    n_none = sum(1 for i in idx_tuple if i is None)
    n_ellipsis = sum(1 for i in idx_tuple if i is ...)
    n_real = len(idx_tuple) - n_none - n_ellipsis
    ellipsis_len = max(ndim - n_real, 0)

    # expand Ellipsis
    expanded: list[Any] = []
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
            length = impl.shape[dim]
            normalized = i + length if i < 0 else i
            if normalized < 0 or normalized >= length:
                raise IndexError(
                    f"index {i} is out of bounds for dimension {dim} "
                    f"with size {length}"
                )
            impl = _C_engine.squeeze(
                _C_engine.slice(impl, dim, normalized, normalized + 1, 1), dim
            )
            # dim does NOT advance — select removed this dimension
        elif isinstance(i, slice):
            length = impl.shape[dim]
            start, stop, step = i.indices(length)
            impl = _C_engine.slice(impl, dim, start, stop, step)
            dim += 1
        else:
            raise IndexError(f"Unsupported index type: {type(i).__name__}")

    return impl


def _setitem(t: Tensor, idx: Any, value: Any) -> None:
    """In-place assignment: t[idx] = value."""
    from lucid._dispatch import _unwrap
    from lucid._tensor.tensor import Tensor as _Tensor

    if isinstance(value, _Tensor):
        v_impl = value._impl
    elif isinstance(value, (int, float, bool)):
        v_impl = _C_engine.full(
            t._impl.shape, float(value), t._impl.dtype, t._impl.device
        )
    else:
        raise TypeError(f"Cannot assign value of type {type(value).__name__}")

    # Use engine's masked_fill or index-based assignment
    # For now, delegate to engine if available, otherwise raise
    if hasattr(_C_engine, "index_put_"):
        _C_engine.index_put_(t._impl, idx, v_impl)
    else:
        raise NotImplementedError(
            "Tensor __setitem__ requires engine.index_put_ support"
        )
