"""Index-based write and scatter operations.

All ops here follow the reference-framework API surface:

* ``index_fill``  — fill elements at 1-D index positions with a scalar.
* ``index_add``   — accumulate scaled source into input at 1-D index positions.
* ``index_copy``  — copy source into input at 1-D index positions.
* ``scatter_reduce`` — scatter-reduce src into input (sum / mean / prod / amax / amin).
* ``masked_scatter`` — copy source elements into positions where mask is True.

All implementations use only engine primitives — no numpy at the Python level.
"""

from typing import TYPE_CHECKING

import lucid
from lucid._dispatch import _unwrap, _wrap
import lucid._C.engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# ── helpers ────────────────────────────────────────────────────────────────


def _to_i32(impl: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """Cast an engine index tensor to ``int32`` if it is not already.

    Used internally by the scatter / gather composites because the engine
    indexing primitives expect ``int32`` index buffers.
    """
    if impl.dtype == _C_engine.I64:
        return _C_engine.astype(impl, _C_engine.I32)
    if impl.dtype != _C_engine.I32:
        return _C_engine.astype(impl, _C_engine.I32)
    return impl


def _dim_indicator(
    size: int,
    positions_impl: _C_engine.TensorImpl,
    device: _C_engine.Device,
) -> _C_engine.TensorImpl:
    """1-D float F32 indicator of length *size*; 1.0 at each listed position."""
    n = int(positions_impl.shape[0]) if positions_impl.shape else 0
    zeros = _C_engine.zeros([size], _C_engine.F32, device)
    if n == 0:
        return zeros
    ones = _C_engine.full([n], 1.0, _C_engine.F32, device)
    idx32 = _to_i32(_C_engine.reshape(positions_impl, [-1]))
    return _C_engine.scatter_add(zeros, idx32, ones, 0)


# ── public API ─────────────────────────────────────────────────────────────


def index_fill(
    input: Tensor,
    dim: int,
    index: Tensor,
    value: float,
) -> Tensor:
    """Return a copy of ``input`` with positions ``index`` along ``dim`` set to ``value``.

    Autograd flows through the *unmasked* positions; filled positions
    receive zero gradient (they're overwritten by a constant).

    Parameters
    ----------
    input : Tensor
        Source tensor; not mutated.
    dim : int
        Axis along which ``index`` addresses slices.
    index : Tensor
        1-D integer tensor of positions along ``dim``.
    value : float
        Scalar to write into every indexed position.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``; indexed slices replaced by
        ``value``, others unchanged.
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


def index_add(
    input: Tensor,
    dim: int,
    index: Tensor,
    source: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Return ``input`` with ``alpha * source`` accumulated at ``index`` positions along ``dim``.

    Differentiable through both ``input`` and ``source``.  ``alpha`` is
    accumulated as a Python constant — gradients pass through cleanly
    as if the multiplication were inlined.

    Parameters
    ----------
    input : Tensor
        Source tensor; not mutated.
    dim : int
        Axis along which ``index`` addresses slices.
    index : Tensor
        1-D integer tensor of length :math:`m` listing positions along
        ``dim`` to accumulate into.
    source : Tensor
        Per-slice update tensor; same shape as ``input`` except
        ``source.shape[dim] == m`` (matching ``index`` length).
    alpha : float, optional
        Scalar multiplier applied to ``source`` before accumulation.
        Default ``1.0``.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``; positions listed in
        ``index`` carry ``input[..., index[i], ...] + alpha * source[..., i, ...]``.
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


def index_copy(
    input: Tensor,
    dim: int,
    index: Tensor,
    source: Tensor,
) -> Tensor:
    """Return a copy of ``input`` with slices at ``index`` replaced by ``source``.

    Single set-scatter: the 1-D ``index`` is broadcast along ``dim`` to
    ``source``'s shape and applied through the engine ``scatter_set`` primitive
    — one MPSGraph ``scatterAlongAxis`` (Set mode) op when compiled, or
    ``mlx.put_along_axis`` eager on GPU.  Differentiable through both ``input``
    and ``source``.

    Parameters
    ----------
    input : Tensor
        Destination tensor; not mutated (a fresh copy is returned).
    dim : int
        Axis along which slices are addressed.
    index : Tensor
        1-D ``int32`` / ``int64`` tensor of positions along ``dim``.
        Length must equal ``source.shape[dim]``.
    source : Tensor
        Replacement slices.  All non-``dim`` dimensions must match
        ``input``; ``source.shape[dim]`` must equal ``index.shape[0]``.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``; values at the indexed
        positions are taken from ``source``, others from ``input``.
    """
    ndim = input.ndim
    if dim < 0:
        dim += ndim
    m = int(source.shape[dim])
    # Broadcast the 1-D index to source's shape, then a single set-scatter.
    idx_impl = _to_i32(_unwrap(index))
    reshaped = [1] * ndim
    reshaped[dim] = m
    idx_bc = _C_engine.broadcast_to(
        _C_engine.reshape(idx_impl, reshaped), list(source.shape)
    )
    return _wrap(_C_engine.scatter_set(_unwrap(input), idx_bc, _unwrap(source), dim))


def scatter_reduce(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    reduce: str = "sum",
    include_self: bool = True,
) -> Tensor:
    """Reduce ``src`` into ``input`` along ``dim`` at positions given by ``index``.

    Multi-reduction sibling of :func:`scatter_add`.  When several entries
    of ``src`` target the same position the chosen ``reduce`` op decides
    how they combine.  ``include_self`` controls whether the existing
    value in ``input`` participates in the reduction or is replaced.

    Parameters
    ----------
    input : Tensor
        Destination tensor; not mutated (a fresh copy is returned).
    dim : int
        Axis along which ``index`` / ``src`` are scattered.
    index : Tensor
        Integer tensor broadcasting against ``src``; each entry names
        the position along ``dim`` of ``input`` to update.
    src : Tensor
        Values to scatter into ``input`` at the positions named by
        ``index``.
    reduce : str, optional
        Reduction op applied when multiple ``src`` values collide on
        the same target.  One of ``'sum'`` (default), ``'mean'``,
        ``'prod'``, ``'amax'``, ``'amin'``.
    include_self : bool, optional
        When ``True`` (default) the existing value in ``input`` is part
        of the reduction set; when ``False`` it is overwritten and only
        the scattered values count.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``.
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


def masked_scatter(input: Tensor, mask: Tensor, source: Tensor) -> Tensor:
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


def index_put(
    input: Tensor,
    indices: list[Tensor] | tuple[Tensor, ...],
    values: Tensor,
    accumulate: bool = False,
) -> Tensor:
    """Out-of-place advanced-indexing write.

    Equivalent to ``out = input.clone(); out[indices] = values`` (or
    ``out[indices] += values`` when ``accumulate=True``) under reference
    framework semantics.  ``indices`` is a sequence of integer tensors,
    one per leading dimension; broadcasting between them follows the
    standard rules.

    Currently restricted to the case where ``len(indices) ==
    input.ndim`` and every index tensor broadcasts to a common shape —
    partial advanced indexing (where trailing dims are implicitly
    sliced) is filed as a follow-up.

    Parameters
    ----------
    input : Tensor
        Destination tensor.
    indices : sequence of Tensors
        One integer index tensor per dimension of ``input``.  All
        broadcast to a common shape.
    values : Tensor
        Values to scatter, broadcastable to the common index shape.
    accumulate : bool, default False
        If True, add at each position; otherwise overwrite.
    """
    if not isinstance(indices, (list, tuple)) or len(indices) == 0:
        raise ValueError("index_put: `indices` must be a non-empty sequence of Tensors")
    if len(indices) != input.ndim:
        raise NotImplementedError(
            f"index_put: partial advanced indexing not supported — "
            f"expected exactly {input.ndim} index tensors, got {len(indices)}"
        )

    # Broadcast all index tensors to a common shape.
    common_shape: tuple[int, ...] = tuple(indices[0].shape)
    for idx in indices[1:]:
        common_shape = (
            lucid._tensor.tensor.broadcast_shapes(common_shape, tuple(idx.shape))
            if hasattr(lucid._tensor, "tensor")
            and hasattr(lucid._tensor.tensor, "broadcast_shapes")
            else common_shape
        )

    bcast_indices: list[Tensor] = []
    for idx in indices:
        if tuple(idx.shape) != common_shape:
            zero = lucid.zeros(common_shape, dtype=idx.dtype, device=idx.device)
            bcast_indices.append(idx + zero)
        else:
            bcast_indices.append(idx)

    # Compute flat indices via multi-dim row-major contraction.
    shape: tuple[int, ...] = tuple(int(s) for s in input.shape)
    strides: list[int] = []
    s: int = 1
    for d in reversed(range(len(shape))):
        strides.insert(0, s)
        s *= shape[d]

    flat_idx: Tensor | None = None
    for d, idx in enumerate(bcast_indices):
        contrib = idx * strides[d]
        flat_idx = contrib if flat_idx is None else flat_idx + contrib
    assert flat_idx is not None

    # Broadcast values to common_shape if scalar/smaller.
    if tuple(values.shape) != common_shape:
        zero = lucid.zeros(common_shape, dtype=values.dtype, device=values.device)
        values_b = values + zero
    else:
        values_b = values

    return put(input, flat_idx, values_b, accumulate=accumulate)


def put(
    input: Tensor,
    index: Tensor,
    source: Tensor,
    accumulate: bool = False,
) -> Tensor:
    """Write ``source`` into ``input`` at the *flat* positions in ``index``.

    Mirrors the reference framework's ``Tensor.put`` semantics: indices
    refer to the row-major linearisation of ``input``, regardless of its
    shape.  ``accumulate=True`` performs additive scatter (duplicates
    add), otherwise duplicates resolve to the last write (``scatter``
    semantics).

    Parameters
    ----------
    input : Tensor
        Destination — its shape is preserved in the output.
    index : Tensor
        1-D (or flattenable) integer tensor of flat positions in
        ``[0, input.numel())``.
    source : Tensor
        Values to scatter; must be flattenable to the same length as
        ``index``.
    accumulate : bool, default False
        If True, add to the existing value at each position; otherwise
        overwrite.
    """
    flat_input: Tensor = input.reshape(-1)
    n: int = int(index.numel())
    flat_index: Tensor = index.reshape(-1)
    flat_source: Tensor = source.reshape(-1).narrow(0, 0, n)

    flat_idx32: Tensor = flat_index.int()
    if accumulate:
        result_flat: Tensor = index_add(flat_input, 0, flat_idx32, flat_source)
    else:
        result_flat = index_copy(flat_input, 0, flat_idx32, flat_source)
    return result_flat.reshape(input.shape)


def index_put_(
    input: Tensor,
    indices: list[Tensor] | tuple[Tensor, ...],
    values: Tensor,
    accumulate: bool = False,
) -> Tensor:
    """In-place variant of :func:`index_put` — mutates ``input`` so the
    write is visible through the same Tensor reference.

    Internally this rebinds ``input._impl`` to the freshly-built result,
    matching the convention Lucid already uses for engine-level in-place
    ops (``add_``, ``mul_``).  Storage-level mutation is not currently
    available for composite indexing — autograd consumers should treat
    the returned tensor as a new node.

    Parameters
    ----------
    input : Tensor
        Destination; mutated in place via ``_impl`` rebind.
    indices : list of Tensor or tuple of Tensor
        Per-axis index tensors (one per dimension of ``input``); same
        contract as :func:`index_put`.
    values : Tensor
        Values to write at the addressed positions.
    accumulate : bool, optional
        When ``True`` add to existing values (duplicate indices sum);
        when ``False`` (default) overwrite (duplicate indices resolve
        to the last write).

    Returns
    -------
    Tensor
        The same ``input`` tensor, now holding the updated values.
    """
    new_t: Tensor = index_put(input, indices, values, accumulate=accumulate)
    input._impl = new_t._impl
    return input


def argwhere(x: Tensor) -> Tensor:
    """Return the coordinates of every non-zero element in ``x``.

    Thin alias for :func:`lucid.nonzero` named to match the NumPy /
    reference-framework convention.  The output is a synchronisation
    point on GPU streams — the kernel can't know how many non-zeros
    there are without a device→host count.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape and dtype; zero is determined by the
        usual truthiness rules (``0`` for numeric dtypes, ``False`` for
        bool).

    Returns
    -------
    Tensor
        Shape ``(N, x.ndim)`` ``int64`` tensor; row ``i`` lists the
        multi-dimensional index of the ``i``-th non-zero element in
        row-major order.

    See Also
    --------
    lucid.nonzero : the underlying engine call.
    """
    return lucid.nonzero(x)


__all__ = [
    "index_fill",
    "index_add",
    "index_copy",
    "scatter_reduce",
    "masked_scatter",
    "put",
    "index_put",
    "index_put_",
    "argwhere",
]
