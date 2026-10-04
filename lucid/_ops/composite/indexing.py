"""Index-based write and scatter operations.

All ops here follow the reference-framework API surface:

* ``index_fill``  — fill elements at 1-D index positions with a scalar.
* ``index_add``   — accumulate scaled source into input at 1-D index positions.
* ``index_copy``  — copy source into input at 1-D index positions.
* ``scatter_reduce`` — scatter-reduce src into input (sum / mean / prod / amax / amin).
* ``masked_scatter`` — copy source elements into positions where mask is True.

All implementations use only engine primitives — no numpy at the Python level.

The in-place forms (``Tensor.index_add_`` and the rest, and
:func:`index_put_`) run the out-of-place op and write its result into the
destination through :func:`_write_inplace`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, final, override

import lucid
from lucid._dispatch import _unwrap, _wrap
import lucid._C.engine as _C_engine
from lucid.autograd.function import Function, FunctionCtx

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# ── helpers ────────────────────────────────────────────────────────────────


@final
class _Snapshot(Function):
    """``x``'s values in a buffer of their own, standing in for ``x`` in the graph.

    What an in-place index op reads its destination through while autograd
    records.  An engine op applied to the destination itself records its
    version, and the write that follows bumps it, so the op's own backward
    refused the write as an in-place modification of a tensor it saved.
    This node saves nothing, and its derivative is the identity.
    """

    @override
    @staticmethod
    def forward(  # type: ignore[override]  # narrower signature than Function by design
        ctx: FunctionCtx, x: Tensor
    ) -> Tensor:
        return x.detach().clone()

    @override
    @staticmethod
    def backward(  # type: ignore[override]  # narrower signature than Function by design
        ctx: FunctionCtx, grad: Tensor
    ) -> Tensor:
        return grad


def _like_input(source: Tensor, input: Tensor) -> Tensor:
    """``source`` in ``input``'s dtype.

    The engine's scatter kernels read ``source``'s buffer as if it held
    ``input``'s dtype: a float64 source added into a float32 tensor landed
    as 1.875 where 1.0 was meant, and an int64 one as 1e-45.  Cast first,
    so the result keeps ``input``'s dtype, as a write into ``input`` must.
    """
    if source._impl.dtype == input._impl.dtype:
        return source
    return source.to(input.dtype)


def _write_inplace[T: Tensor](
    input: T, name: str, result_of: Callable[[Tensor], Tensor], *operands: Tensor
) -> T:
    """Run an out-of-place index op and make its result ``input``'s values.

    A leaf that requires grad is refused while autograd records, as every
    in-place op refuses it; under :func:`lucid.no_grad` it is written and
    stays a trainable leaf.  Otherwise the result lands one of three ways,
    the ones ``Tensor.__setitem__`` takes:

    * ``input`` shares its buffer with a live view: the values go into the
      buffer, where the views read them, and the engine moves ``input``
      and its views to the result's place in the graph;
    * autograd records: ``input`` takes the result's tensor, and with it
      the result's place in the graph.  Copied into ``input``'s buffer
      instead, a tensor that only *received* a gradient-carrying ``source``
      kept its leaf flag, and every later in-place op refused it;
    * otherwise the values are copied into ``input``'s buffer.

    In the first case, while autograd records, the op reads ``input``
    through a :class:`_Snapshot`.  Its nodes save what they read — ``where``
    does — and the write bumps the buffer's version, so read directly,
    ``input`` made backward refuse (``VersionMismatch``) the very op that
    wrote it.

    Parameters
    ----------
    input : Tensor
        The destination, written in place and returned.
    name : str
        The in-place op's name, for errors.
    result_of : callable
        The out-of-place op, given the tensor to read ``input`` from.
    *operands : Tensor
        The op's other differentiable inputs (``source``, ``values``).

    Returns
    -------
    Tensor
        ``input``, now holding the result.

    Raises
    ------
    RuntimeError
        If ``input`` is a leaf that requires grad and autograd is recording.
    """
    records = _C_engine.grad_enabled() and (
        input.requires_grad or any(o.requires_grad for o in operands)
    )
    if records and input.requires_grad and input.is_leaf:
        raise RuntimeError(
            f"{name}: a leaf tensor that requires grad cannot be modified in place — "
            "wrap the call in lucid.no_grad(), or use the out-of-place form"
        )
    aliased = input._impl.is_aliased()
    base: Tensor = input
    if records and aliased:
        snapshot = _Snapshot.apply(input)
        assert isinstance(snapshot, lucid.Tensor)
        base = snapshot
    result = result_of(base)
    if aliased:
        _C_engine.assign_inplace(input._impl, result._impl, name)
    elif records:
        input._impl = result._impl
    else:
        input._impl.assign_from(result._impl, name)
    return input


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
        ``source.shape[dim] == m`` (matching ``index`` length).  Cast to
        ``input``'s dtype first.
    alpha : float, optional
        Scalar multiplier applied to ``source`` before accumulation.
        Default ``1.0``.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``; positions listed in
        ``index`` carry ``input[..., index[i], ...] + alpha * source[..., i, ...]``.
    """
    source = _like_input(source, input)
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
        Cast to ``input``'s dtype first.

    Returns
    -------
    Tensor
        Same shape and dtype as ``input``; values at the indexed
        positions are taken from ``source``, others from ``input``.
    """
    source = _like_input(source, input)
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

    Fewer index tensors than dimensions index the leading dimensions and
    take the rest whole, as the reference does: ``index_put(x, (i,), v)``
    on a ``(4, 3)`` tensor writes whole rows.

    Parameters
    ----------
    input : Tensor
        Destination tensor.
    indices : sequence of Tensors
        One integer index tensor per leading dimension of ``input``; the
        dimensions after them are taken whole.  All broadcast to a common
        shape.
    values : Tensor
        Values to scatter, broadcastable to the common index shape
        followed by the dimensions taken whole.
    accumulate : bool, default False
        If True, add at each position; otherwise overwrite.
    """
    if not isinstance(indices, (list, tuple)) or len(indices) == 0:
        raise ValueError("index_put: `indices` must be a non-empty sequence of Tensors")
    if len(indices) > input.ndim:
        raise IndexError(
            f"index_put: too many indices for a {input.ndim}-D tensor: "
            f"got {len(indices)}"
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

    # The dimensions after the indexed ones are taken whole.  They are
    # contiguous in row-major order, so each index selects a run of flat
    # positions: the block's start plus 0, 1, ..., block - 1.
    block_shape = shape[len(indices) :]
    block = 1
    for extent in block_shape:
        block *= extent
    if block_shape:
        offsets = lucid.arange(block, dtype=flat_idx.dtype, device=flat_idx.device)
        flat_idx = flat_idx.reshape(*common_shape, 1) + offsets
    target_shape = common_shape + block_shape

    # Broadcast values to the indexed shape if scalar/smaller.
    if tuple(values.shape) != target_shape:
        zero = lucid.zeros(target_shape, dtype=values.dtype, device=values.device)
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
    """In-place variant of :func:`index_put`: writes into ``input``'s storage.

    The values land in ``input``'s own buffer, so a view of ``input`` (or
    the tensor ``input`` is a view of) sees them, and ``input`` takes the
    write's place in the autograd graph, as the engine's in-place ops do.
    A leaf that requires grad is refused while autograd records — wrap the
    write in :func:`lucid.no_grad`, which leaves a Parameter a trainable
    leaf.

    Parameters
    ----------
    input : Tensor
        Destination; written in place.
    indices : list of Tensor or tuple of Tensor
        Per-axis index tensors (one per leading dimension of ``input``);
        same contract as :func:`index_put`.
    values : Tensor
        Values to write at the addressed positions; cast to ``input``'s
        dtype.
    accumulate : bool, optional
        When ``True`` add to existing values (duplicate indices sum);
        when ``False`` (default) overwrite (duplicate indices resolve
        to the last write).

    Returns
    -------
    Tensor
        The same ``input`` tensor, now holding the updated values.

    Raises
    ------
    RuntimeError
        If ``input`` is a leaf that requires grad and autograd is recording.
    """
    return _write_inplace(
        input,
        "index_put_",
        lambda base: index_put(base, indices, values, accumulate=accumulate),
        values,
    )


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
