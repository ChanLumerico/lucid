"""Index-based write and scatter operations.

All ops here follow the PyTorch API surface:

* ``index_fill``  — fill elements at 1-D index positions with a scalar.
* ``index_add``   — accumulate scaled source into input at 1-D index positions.
* ``index_copy``  — copy source into input at 1-D index positions.
* ``scatter_reduce`` — scatter-reduce src into input (sum / mean / prod / amax / amin).
* ``masked_scatter`` — copy source elements into positions where mask is True.

All implementations use only engine primitives — no numpy at the Python level.
"""

import lucid
from lucid._dispatch import _unwrap, _wrap
import lucid._C.engine as _C_engine

# ── helpers ────────────────────────────────────────────────────────────────


def _to_i32(impl: "_C_engine.TensorImpl") -> "_C_engine.TensorImpl":
    if impl.dtype == _C_engine.I64:
        return _C_engine.astype(impl, _C_engine.I32)
    if impl.dtype != _C_engine.I32:
        return _C_engine.astype(impl, _C_engine.I32)
    return impl


def _dim_indicator(
    size: int,
    positions_impl: "_C_engine.TensorImpl",
    device,
) -> "_C_engine.TensorImpl":
    """1-D float F32 indicator of length *size*; 1.0 at each listed position."""
    n = int(positions_impl.shape[0]) if positions_impl.shape else 0
    zeros = _C_engine.zeros([size], _C_engine.F32, device)
    if n == 0:
        return zeros
    ones = _C_engine.full([n], 1.0, _C_engine.F32, device)
    idx32 = _to_i32(_C_engine.reshape(positions_impl, [-1]))
    return _C_engine.scatter_add(zeros, idx32, ones, 0)


# ── public API ─────────────────────────────────────────────────────────────


def index_fill(input, dim: int, index, value: float):
    """Return a copy of ``input`` with positions ``index`` along ``dim`` set to ``value``.

    Autograd flows through the *unmasked* positions; filled positions receive
    zero gradient (they are overwritten by a constant).
    """
    ndim = input.ndim
    if dim < 0:
        dim += ndim
    n = input.shape[dim]
    device = input._impl.device

    idx_impl = _to_i32(_unwrap(index))
    indicator = _dim_indicator(n, idx_impl, device)

    bcast_shape = [1] * ndim
    bcast_shape[dim] = n
    mask = _wrap(
        _C_engine.broadcast_to(
            _C_engine.reshape(indicator, bcast_shape), list(input.shape)
        )
    )

    return lucid.where(mask > 0.0, lucid.full_like(input, float(value)), input)


def index_add(input, dim: int, index, source, alpha: float = 1.0):
    """Return ``input`` with ``alpha * source`` accumulated at ``index`` positions along ``dim``.

    ``index`` is a 1-D integer tensor of length *m*;
    ``source`` has the same shape as ``input`` except ``source.shape[dim] == m``.
    """
    ndim = input.ndim
    if dim < 0:
        dim += ndim
    m = int(source.shape[dim])

    # Reshape the 1-D index to broadcast along dim, then expand to source.shape.
    idx_impl = _to_i32(_unwrap(index))
    rs = [1] * ndim
    rs[dim] = m
    idx_rs = _C_engine.reshape(idx_impl, rs)
    idx_bc = _C_engine.broadcast_to(idx_rs, list(source.shape))
    idx_t = _wrap(idx_bc)

    scaled = source * float(alpha) if alpha != 1.0 else source
    return input.scatter_add(dim, idx_t, scaled)


def index_copy(input, dim: int, index, source):
    """Return a copy of ``input`` with slices at ``index`` replaced by ``source``."""
    zeroed = index_fill(input, dim, index, 0.0)
    return index_add(zeroed, dim, index, source)


def scatter_reduce(
    input, dim: int, index, src, reduce: str = "sum", include_self: bool = True
):
    """Reduce ``src`` into ``input`` along ``dim`` at positions given by ``index``.

    Supported ``reduce`` modes: ``'sum'``, ``'mean'``, ``'prod'``, ``'amax'``, ``'amin'``.
    """
    if reduce == "sum":
        base = input if include_self else lucid.zeros_like(input)
        return base.scatter_add(dim, index, src)

    elif reduce == "mean":
        ones = lucid.ones_like(src)
        count = lucid.zeros_like(input).scatter_add(dim, index, ones)
        base = input if include_self else lucid.zeros_like(input)
        total = base.scatter_add(dim, index, src)
        denom = count + (1.0 if include_self else 0.0)
        safe_denom = lucid.where(denom > 0.0, denom, lucid.ones_like(denom))
        default = input if include_self else lucid.zeros_like(input)
        return lucid.where(denom > 0.0, total / safe_denom, default)

    elif reduce in ("amax", "amin", "prod"):
        _NEUTRAL = {
            "amax": float("-inf"),
            "amin": float("inf"),
            "prod": 1.0,
        }
        base = input if include_self else lucid.full_like(input, _NEUTRAL[reduce])

        # Coerce index to int32 (engine scatter kernels require int32).
        idx_impl = _unwrap(index)
        idx_i32 = _wrap(_to_i32(idx_impl))

        base_impl = _unwrap(base)
        idx_impl_i32 = _unwrap(idx_i32)
        src_impl = _unwrap(src)

        _fn = {
            "amax": _C_engine.scatter_amax,
            "amin": _C_engine.scatter_amin,
            "prod": _C_engine.scatter_prod,
        }[reduce]

        from lucid._tensor.tensor import Tensor as _Tensor

        out_impl = _fn(base_impl, idx_impl_i32, src_impl, dim)
        out = _Tensor.__new_from_impl__(out_impl)

        if not include_self:
            ones = lucid.ones_like(src)
            count = lucid.zeros_like(input).scatter_add(dim, idx_i32, ones)
            out = lucid.where(count > 0.0, out, input)

        return out

    else:
        raise ValueError(
            f"scatter_reduce: unknown reduce={reduce!r}; "
            "expected 'sum', 'mean', 'prod', 'amax', or 'amin'."
        )


def masked_scatter(input, mask, source):
    """Copy elements from ``source`` into ``input`` at positions where ``mask`` is True."""
    flat_input = input.reshape(-1)
    flat_mask = mask.reshape(-1)

    true_idx = lucid.nonzero(flat_mask)  # (n_true, 1)
    n_true = int(true_idx.shape[0])
    if n_true == 0:
        return input

    true_idx_1d = true_idx.squeeze(1).int()  # (n_true,) int32
    src_vals = source.reshape(-1).narrow(0, 0, n_true)

    result_flat = index_copy(flat_input, 0, true_idx_1d, src_vals)
    return result_flat.reshape(input.shape)


__all__ = [
    "index_fill",
    "index_add",
    "index_copy",
    "scatter_reduce",
    "masked_scatter",
]
