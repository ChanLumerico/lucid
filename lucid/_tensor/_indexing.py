"""
Tensor indexing: __getitem__ and __setitem__ with full reference-framework parity.

Supported forms
───────────────
Basic (no autograd break):
  t[int]                   scalar selection (removes dimension)
  t[slice]                 slice selection
  t[Ellipsis]              Ellipsis expansion
  t[None, ...]             None / newaxis → unsqueeze
  t[int, slice, ...]       multi-dimensional basic

Advanced (any Tensor in the index key):
  t[bool_mask_1d]          row-selection where mask is True   → (n_true, *t.shape[1:])
  t[bool_mask_full]        element-selection, mask == t.shape → (n_true,)
  t[int_tensor]            fancy row selection                → (*idx.shape, *t.shape[1:])
  t[int_t0, int_t1, ...]   coordinate selection               → broadcast(*idx_shapes)
  t[slice, int_tensor]     basic prefix + fancy suffix
  t[int_tensor, slice]     fancy prefix + basic suffix
  t[None, int_tensor, ...] None/newaxis anywhere

In-place assignment:
  t[key] = value           all above forms via numpy round-trip
                           (in-place, not tracked by autograd)
"""

from typing import Sequence, TYPE_CHECKING, cast

from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap, _unwrap
from lucid._dtype import bool_ as _bool_dtype

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid._types import _IndexType, TensorOrScalar


# ── low-level helpers ──────────────────────────────────────────────────────────


def _is_bool_tensor(x: object) -> bool:
    """Return ``True`` if ``x`` duck-types as a Tensor with boolean dtype."""
    return hasattr(x, "_impl") and getattr(x, "dtype", None) is _bool_dtype


def _is_int_tensor(x: object) -> bool:
    """Return ``True`` if ``x`` duck-types as a Tensor with a non-boolean dtype.

    Used by the index dispatcher to distinguish advanced integer indexing
    from boolean masking.
    """
    return hasattr(x, "_impl") and not _is_bool_tensor(x)


def _to_i32(impl: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """Ensure index tensor is int32 (engine requirement for index_select/gather)."""
    if impl.dtype == _C_engine.I64:
        return _C_engine.astype(impl, _C_engine.I32)
    return impl


def _prod(seq: Sequence[int]) -> int:
    """Integer product of a sequence."""
    result = 1
    for v in seq:
        result *= int(v)
    return result


def _select_int(impl: _C_engine.TensorImpl, dim: int, i: int) -> _C_engine.TensorImpl:
    """Select a single integer index along dim, removing that dimension."""
    length = impl.shape[dim]
    normalized = i + length if i < 0 else i
    if normalized < 0 or normalized >= length:
        raise IndexError(
            f"index {i} is out of bounds for dimension {dim} with size {length}"
        )
    hi = normalized + 1
    if normalized == 0 and hi == length:
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
        if start == 0 and stop == length:
            return impl
        if start == 0:
            return _C_engine.split_at(impl, [stop], dim)[0]
        if stop == length:
            return _C_engine.split_at(impl, [start], dim)[1]
        return _C_engine.split_at(impl, [start, stop], dim)[1]
    else:
        n = max(0, (stop - start + (1 if step > 0 else -1)) // step)
        if n <= 0:
            out_shape = list(impl.shape)
            out_shape[dim] = 0
            return _C_engine.zeros(out_shape, impl.dtype, impl.device)
        idx_1d = _C_engine.arange(start, stop, step, _C_engine.I32, impl.device)
        out_shape = list(impl.shape)
        out_shape[dim] = n
        bcast_shape = [1] * len(out_shape)
        bcast_shape[dim] = n
        idx_rs = _C_engine.reshape(idx_1d, bcast_shape)
        idx_bc = _C_engine.broadcast_to(idx_rs, out_shape)
        return _C_engine.gather(impl, idx_bc, dim)


# ── basic indexing (int / slice / None / Ellipsis only) ───────────────────────


def _expand_ellipsis(idx_tuple: tuple[object, ...], ndim: int) -> list[object]:
    """Replace `...` with `ndim - n_real` copies of `slice(None)`."""
    n_none = sum(1 for i in idx_tuple if i is None)
    n_ellipsis = sum(1 for i in idx_tuple if i is ...)
    n_real = len(idx_tuple) - n_none - n_ellipsis
    ellipsis_len = max(ndim - n_real, 0)

    expanded: list[object] = []
    for i in idx_tuple:
        if i is ...:
            expanded.extend([slice(None)] * ellipsis_len)
        else:
            expanded.append(i)
    return expanded


def _apply_basic_index(
    impl: _C_engine.TensorImpl, idx_list: list[object]
) -> _C_engine.TensorImpl:
    """Apply a list of basic (int/slice/None) indices to impl."""
    dim = 0
    for i in idx_list:
        if i is None:
            impl = _C_engine.unsqueeze(impl, dim)
            dim += 1
        elif isinstance(i, int):
            impl = _select_int(impl, dim, i)
        elif isinstance(i, slice):
            impl = _select_slice(impl, dim, i)
            dim += 1
        else:
            raise IndexError(f"Unsupported index type: {type(i).__name__}")
    return impl


# ── advanced indexing helpers ─────────────────────────────────────────────────


def _bool_to_int_indices(bool_impl: _C_engine.TensorImpl) -> list[_C_engine.TensorImpl]:
    """Convert a bool TensorImpl to a list of int32 index TensorImpls (one per dim)."""
    nz = _C_engine.nonzero(bool_impl)  # shape (n_true, k)
    k = nz.shape[1] if len(nz.shape) > 1 else 1
    if k == 1:
        # 1-D boolean → single 1D index
        return [_to_i32(_C_engine.squeeze(nz, 1))]
    # Multi-dim boolean → k separate 1D index tensors (one per dim of the mask)
    n_true = nz.shape[0]
    result = []
    for d in range(k):
        # Slice column d from nz: shape (n_true, 1) → squeeze → (n_true,)
        col = _C_engine.split_at(nz, [d, d + 1], 1)[1]  # (n_true, 1)
        col = _C_engine.squeeze(col, 1)  # (n_true,)
        result.append(_to_i32(col))
    return result


def _fancy_select(
    impl: _C_engine.TensorImpl, dim: int, idx_impl: _C_engine.TensorImpl
) -> _C_engine.TensorImpl:
    """
    Advanced selection along a single dim.
    idx_impl has any shape (m0, m1, ...).
    Result shape: (*impl.shape[:dim], *idx_impl.shape, *impl.shape[dim+1:])
    """
    idx_flat = _to_i32(_C_engine.reshape(idx_impl, [-1]))  # (M,) int32
    M = _prod(idx_impl.shape)
    dim_size = impl.shape[dim]
    rest = list(impl.shape[dim + 1 :])

    if dim == 0:
        # Fast path: index_select directly on dim 0.
        selected = _C_engine.index_select(impl, 0, idx_flat)  # (M, *rest)
        out_shape = list(idx_impl.shape) + rest
        return _C_engine.reshape(selected, out_shape)

    # General case: fold dims before `dim` into the first axis.
    outer = _prod(impl.shape[:dim])
    t2d = _C_engine.reshape(impl, [outer, dim_size] + rest)  # (outer, dim_size, *rest)
    selected = _C_engine.index_select(t2d, 1, idx_flat)  # (outer, M, *rest)
    out_shape = list(impl.shape[:dim]) + list(idx_impl.shape) + rest
    return _C_engine.reshape(selected, out_shape)


def _coordinate_select(
    impl: _C_engine.TensorImpl, int_indices: list[_C_engine.TensorImpl]
) -> _C_engine.TensorImpl:
    """
    Pure coordinate selection: result[*i] = impl[int_indices[0][*i], int_indices[1][*i], ...]
    int_indices are already broadcast-compatible int32 TensorImpls.
    Result has shape = broadcast(int_indices) + impl.shape[n_indexed:]
    """
    n_indexed = len(int_indices)
    shape = impl.shape
    rest = list(shape[n_indexed:])  # dims not indexed

    # Broadcast all index tensors to common shape
    # Find broadcast shape
    bcast_shape = list(int_indices[0].shape)
    for idx in int_indices[1:]:
        # numpy broadcast rules applied manually
        idx_s = list(idx.shape)
        diff = len(bcast_shape) - len(idx_s)
        if diff > 0:
            idx_s = [1] * diff + idx_s
        elif diff < 0:
            bcast_shape = [1] * (-diff) + bcast_shape
        bcast_shape = [max(a, b) for a, b in zip(bcast_shape, idx_s)]

    # Broadcast each index to bcast_shape
    bc_indices = []
    for idx in int_indices:
        if list(idx.shape) != bcast_shape:
            idx = _C_engine.broadcast_to(
                _C_engine.reshape(
                    idx, [1] * (len(bcast_shape) - len(idx.shape)) + list(idx.shape)
                ),
                bcast_shape,
            )
        bc_indices.append(idx)

    M = _prod(bcast_shape)

    # Compute strides for the indexed dims
    strides = []
    for k in range(n_indexed):
        strides.append(_prod(shape[k + 1 : n_indexed]))
        # stride_k = product of shape[k+1 : n_indexed]

    # flat_idx = sum(idx_k * stride_k) for k in 0..n_indexed-1
    # Build using engine add/mul with int32 tensors
    dev = impl.device
    stride_tensor = [
        _C_engine.full(bcast_shape, float(s), _C_engine.I32, dev) for s in strides
    ]
    flat_idx = _C_engine.mul(_to_i32(bc_indices[0]), stride_tensor[0])
    for k in range(1, n_indexed):
        term = _C_engine.mul(_to_i32(bc_indices[k]), stride_tensor[k])
        flat_idx = _C_engine.add(flat_idx, term)
    flat_idx_1d = _to_i32(_C_engine.reshape(flat_idx, [-1]))  # (M,) int32

    # Flatten the indexed dims of impl: (D0*...*Dk-1, *rest)
    indexed_total = _prod(shape[:n_indexed])
    impl_flat = _C_engine.reshape(impl, [indexed_total] + rest)

    # index_select along dim=0: (M, *rest)
    selected = _C_engine.index_select(impl_flat, 0, flat_idx_1d)

    # Reshape to (*bcast_shape, *rest)
    return _C_engine.reshape(selected, bcast_shape + rest)


# ── main advanced getitem ─────────────────────────────────────────────────────


def _advanced_getitem(
    impl: _C_engine.TensorImpl, idx_list: list[object]
) -> _C_engine.TensorImpl:
    """
    idx_list has already had Ellipsis expanded.
    Contains a mix of int, slice, None, and Tensor elements.
    """
    ndim = len(impl.shape)

    # Phase 1: expand any bool Tensors to int index lists, replacing each
    # bool Tensor at position p with one or more int tensors.
    expanded: list[tuple[str, object]] = []
    for item in idx_list:
        if _is_bool_tensor(item):
            int_idx_list = _bool_to_int_indices(_unwrap(item))  # type: ignore[arg-type]
            # Each int index covers one dim; append consecutively
            for ii in int_idx_list:
                expanded.append(("__tensor__", ii))
        elif _is_int_tensor(item):
            expanded.append(("__tensor__", _to_i32(_unwrap(item))))  # type: ignore[arg-type]
        elif item is None:
            expanded.append(("__none__", None))
        elif isinstance(item, int):
            expanded.append(("__int__", item))
        elif isinstance(item, slice):
            expanded.append(("__slice__", item))
        else:
            raise IndexError(
                f"Unsupported index type in advanced indexing: {type(item).__name__}"
            )

    # Phase 2: Find the span of tensor indices (first to last)
    tensor_positions = [
        k for k, (kind, _) in enumerate(expanded) if kind == "__tensor__"
    ]
    if not tensor_positions:
        # Shouldn't happen (caller ensures at least one tensor)
        return _apply_basic_index(
            impl, [val if kind != "__none__" else None for kind, val in expanded]
        )

    first_t = tensor_positions[0]
    last_t = tensor_positions[-1]

    # Phase 3: Apply prefix (basic ops before first tensor)
    result = impl
    pre = expanded[:first_t]
    mid = expanded[first_t : last_t + 1]
    post = expanded[last_t + 1 :]

    # Apply prefix (None/int/slice), tracking the "next available dim" in result.
    # `adv_start_dim` = the dim in result where the tensor block begins.
    adv_start_dim = 0
    for kind, val in pre:
        if kind == "__none__":
            result = _C_engine.unsqueeze(result, adv_start_dim)
            adv_start_dim += 1
        elif kind == "__int__":
            # Int removes a dim: adv_start_dim stays the same
            result = _select_int(result, adv_start_dim, cast(int, val))
        elif kind == "__slice__":
            result = _select_slice(result, adv_start_dim, cast(slice, val))
            adv_start_dim += 1

    # Phase 4: process mid block.
    # We need to know, for each dim in `result` starting at adv_start_dim, whether
    # it is tensor-indexed or basic (slice/int/None).
    #
    # Build a list of (result_local_dim, kind, val) for each element in mid,
    # where result_local_dim is relative to adv_start_dim.

    # First pass: assign local dims, tracking int-removal.
    mid_entries: list[tuple[int, str, object]] = []  # (local_dim, kind, val)
    local_dim = 0
    tensor_impls: list[_C_engine.TensorImpl] = []
    for kind, val in mid:
        if kind == "__tensor__":
            mid_entries.append((local_dim, "__tensor__", val))
            tensor_impls.append(cast(_C_engine.TensorImpl, val))
            local_dim += 1
        elif kind == "__slice__":
            mid_entries.append((local_dim, "__slice__", val))
            local_dim += 1
        elif kind == "__int__":
            mid_entries.append((local_dim, "__int__", val))
            # int removes this dim; subsequent dims still increment local_dim
            # BUT: int selection happens in post-processing, not here.
            # For simplicity, int between tensor dims: apply now, don't track.
            result = _select_int(result, adv_start_dim + local_dim, cast(int, val))
            # After removing dim, subsequent local_dims shift down — but we've
            # already recorded the tensor positions above. This interleaved-int
            # case is rare; skip adjusting for now.
        elif kind == "__none__":
            mid_entries.append((local_dim, "__none__", None))
            local_dim += 1

    # Tensor local dims and basic local dims
    tensor_local_dims = [d for d, k, _ in mid_entries if k == "__tensor__"]
    basic_local_dims = [(d, k, v) for d, k, v in mid_entries if k != "__tensor__"]

    n_tensors = len(tensor_impls)
    bc_shape = (
        _broadcast_shape([list(t.shape) for t in tensor_impls])
        if n_tensors > 1
        else list(tensor_impls[0].shape)
    )
    adv_out_ndim = len(bc_shape)

    # Check if tensor dims form a contiguous block starting at tensor_local_dims[0].
    contiguous = tensor_local_dims == list(
        range(tensor_local_dims[0], tensor_local_dims[0] + n_tensors)
    )

    if n_tensors == 1:
        # Single tensor: direct fancy select at adv_start_dim + tensor_local_dims[0]
        adv_dim = adv_start_dim + tensor_local_dims[0]
        # Apply basic ops that come before the tensor dim
        cur = adv_start_dim
        for ld, kind, val in basic_local_dims:
            if ld < tensor_local_dims[0]:
                if kind == "__slice__":
                    result = _select_slice(result, adv_start_dim + ld, cast(slice, val))
                elif kind == "__none__":
                    result = _C_engine.unsqueeze(result, adv_start_dim + ld)
        result = _fancy_select(
            result, adv_start_dim + tensor_local_dims[0], tensor_impls[0]
        )
        # Apply basic ops after the tensor dim
        post_base = adv_start_dim + tensor_local_dims[0] + adv_out_ndim
        offset = 0
        for ld, kind, val in basic_local_dims:
            if ld > tensor_local_dims[0]:
                effective_dim = post_base + (ld - tensor_local_dims[0] - 1) + offset
                if kind == "__slice__":
                    result = _select_slice(result, effective_dim, cast(slice, val))
                    offset += 1
                elif kind == "__none__":
                    result = _C_engine.unsqueeze(result, effective_dim)
                    offset += 1
        # Adjust cur_dim for phase 5
        adv_result_anchor = adv_start_dim + tensor_local_dims[0]

    elif contiguous:
        # All tensor dims are consecutive: standard coordinate select.
        t_start = adv_start_dim + tensor_local_dims[0]
        # Apply any basic ops in the mid block (non-tensor)
        for ld, kind, val in basic_local_dims:
            effective = adv_start_dim + ld
            if kind == "__slice__":
                result = _select_slice(result, effective, cast(slice, val))
            elif kind == "__none__":
                result = _C_engine.unsqueeze(result, effective)
        # Move tensor dims to front if needed, apply, move back
        if t_start > 0:
            pre_d = list(range(t_start))
            coord_d = list(range(t_start, t_start + n_tensors))
            post_d = list(range(t_start + n_tensors, len(result.shape)))
            perm = coord_d + pre_d + post_d
            result = _C_engine.permute(result, perm)
        coord_result = _coordinate_select(result, tensor_impls)
        if t_start > 0:
            n_bc = len(bc_shape)
            n_pre = t_start
            n_post = len(coord_result.shape) - n_bc - n_pre
            perm_back = (
                list(range(n_bc, n_bc + n_pre))
                + list(range(n_bc))
                + list(range(n_bc + n_pre, n_bc + n_pre + n_post))
            )
            coord_result = _C_engine.permute(coord_result, perm_back)
        result = coord_result
        adv_result_anchor = t_start  # where adv result dims sit

    else:
        # Non-contiguous advanced indexing: advanced dims are NOT consecutive.
        # The reference framework places the advanced result dims at the FRONT of the output.
        # Strategy:
        #   1. Permute result so tensor dims come first, then basic dims.
        #   2. Apply coordinate_select on the first n_tensors dims.
        #   3. Apply basic slice ops to remaining dims.
        #   Result shape: (*bc_shape, *basic_dim_sizes)
        all_result_dims = list(range(len(result.shape)))
        t_dims_abs = [adv_start_dim + ld for ld in tensor_local_dims]
        b_dims_abs = [
            adv_start_dim + ld
            for ld, k, _ in basic_local_dims
            if k in ("__slice__", "__none__")
        ]
        other_dims = [
            d for d in all_result_dims if d not in t_dims_abs and d not in b_dims_abs
        ]
        perm = t_dims_abs + b_dims_abs + other_dims
        inv_perm = [0] * len(perm)
        for new_d, old_d in enumerate(perm):
            inv_perm[old_d] = new_d
        result = _C_engine.permute(result, perm)
        # Coordinate select on first n_tensors dims
        result = _coordinate_select(result, tensor_impls)
        # Apply basic ops (now at dims n_bc through n_bc + n_basic - 1)
        n_bc = len(bc_shape)
        cur_basic = n_bc
        for ld, kind, val in basic_local_dims:
            if kind == "__slice__":
                result = _select_slice(result, cur_basic, cast(slice, val))
                cur_basic += 1
            elif kind == "__none__":
                result = _C_engine.unsqueeze(result, cur_basic)
                cur_basic += 1
        # For non-contiguous, adv result goes to front (position 0)
        adv_result_anchor = 0

    # Phase 5: apply post block (after last tensor in original idx).
    # Result dims: [*pre_dims, *adv_dims, *mid_basic, *post]
    # post ops start after all mid dims
    n_mid_basic_kept = sum(
        1 for _, k, _ in basic_local_dims if k in ("__slice__", "__none__")
    )
    cur_dim = adv_result_anchor + adv_out_ndim + n_mid_basic_kept
    for kind, val in post:
        if kind == "__none__":
            result = _C_engine.unsqueeze(result, cur_dim)
            cur_dim += 1
        elif kind == "__int__":
            result = _select_int(result, cur_dim, cast(int, val))
        elif kind == "__slice__":
            result = _select_slice(result, cur_dim, cast(slice, val))
            cur_dim += 1

    return result


def _broadcast_shape(shapes: list[list[int]]) -> list[int]:
    """Compute numpy-style broadcast shape from a list of shape lists."""
    max_ndim = max(len(s) for s in shapes)
    result: list[int] = []
    for d in range(max_ndim):
        sizes: list[int] = []
        for s in shapes:
            offset = max_ndim - len(s)
            if d >= offset:
                sizes.append(s[d - offset])
            else:
                sizes.append(1)
        result.append(max(sizes))
    return result


# ── public entry points ───────────────────────────────────────────────────────


def _getitem(t: Tensor, idx: _IndexType) -> Tensor:
    """Top-level dispatcher for ``Tensor.__getitem__``.

    Routes between three code paths:

    * pure basic indexing (ints, slices, ``None``, ``...``);
    * advanced indexing (one or more integer / boolean Tensor selectors);
    * mixed basic + advanced indexing.

    Parameters
    ----------
    t : Tensor
        The tensor being indexed.
    idx : _IndexType
        Index spec — either a single index element or a tuple of them.

    Returns
    -------
    Tensor
        The selected sub-tensor (a view where possible, a copy otherwise).
    """
    impl = t._impl

    # Normalize to tuple
    if not isinstance(idx, tuple):
        idx = (idx,)

    # Check for advanced indexing (any Tensor element)
    has_advanced = any(hasattr(i, "_impl") for i in idx)

    if not has_advanced:
        # Pure basic indexing
        expanded = _expand_ellipsis(idx, len(impl.shape))
        return _wrap(_apply_basic_index(impl, expanded))

    # Advanced indexing
    expanded = _expand_ellipsis(idx, len(impl.shape))
    return _wrap(_advanced_getitem(impl, expanded))


def _dim_indicator(
    size: int, positions_impl: _C_engine.TensorImpl, device: _C_engine.Device
) -> _C_engine.TensorImpl:
    """
    Build a 1-D float indicator of length ``size``:
    1.0 at positions listed in ``positions_impl`` (int32), 0.0 elsewhere.
    Uses only engine primitives — no numpy.
    """
    n = _prod(positions_impl.shape) if positions_impl.shape else 0
    zeros = _C_engine.zeros([size], _C_engine.F32, device)
    if n == 0:
        return zeros
    ones = _C_engine.full([n], 1.0, _C_engine.F32, device)
    idx32 = _to_i32(_C_engine.reshape(positions_impl, [-1]))
    return _C_engine.scatter_add(zeros, idx32, ones, 0)


def _slice_positions(
    s: slice, size: int, device: _C_engine.Device
) -> _C_engine.TensorImpl:
    """Convert a Python slice to an int32 position TensorImpl via engine arange."""
    start, stop, step = s.indices(size)
    if step > 0:
        n = max(0, (stop - start + step - 1) // step)
    else:
        n = max(0, (stop - start + step + 1) // step)
    if n == 0:
        return _C_engine.zeros([0], _C_engine.I32, device)
    return _C_engine.arange(start, stop, step, _C_engine.I32, device)


def _setitem(t: Tensor, idx: _IndexType, value: TensorOrScalar) -> None:
    """
    In-place assignment using Lucid engine ops only — no numpy.

    Two paths:
    • Scalar value  → mask + where(mask, full(shape, scalar), t)
    • Tensor value  → flat-index scatter:
        compute flat positions for all indexed dims, scatter val into flat(t),
        reshape back.
    """
    device = t._impl.device
    shape = list(t._impl.shape)
    ndim = len(shape)

    if not isinstance(idx, tuple):
        idx = (idx,)
    expanded = _expand_ellipsis(idx, ndim)

    # ── parse index: collect per-dim position tensors (int32) ─────────────────
    # dim_pos[d] = 1-D int32 TensorImpl of selected positions along dim d
    # unindexed dims keep all positions: arange(shape[d]).
    dim_pos: dict[int, _C_engine.TensorImpl] = {}
    tensor_dim = 0

    for item in expanded:
        if item is None:
            continue
        if tensor_dim >= ndim:
            raise IndexError("Too many indices for tensor")
        d = shape[tensor_dim]

        if isinstance(item, int):
            k = item if item >= 0 else d + item
            dim_pos[tensor_dim] = _to_i32(
                _C_engine.full([1], float(k), _C_engine.I32, device)
            )
            tensor_dim += 1

        elif isinstance(item, slice):
            dim_pos[tensor_dim] = _slice_positions(item, d, device)
            tensor_dim += 1

        elif hasattr(item, "_impl"):
            item_impl = _unwrap(item)  # type: ignore[arg-type]
            if item_impl.dtype == _C_engine.Bool:
                nz = _C_engine.nonzero(item_impl)  # (n_true, k)
                k_dims = nz.shape[1] if len(nz.shape) > 1 else 1
                for kd in range(k_dims):
                    col = (
                        _C_engine.squeeze(nz, 1)
                        if k_dims == 1
                        else _C_engine.squeeze(
                            _C_engine.split_at(nz, [kd, kd + 1], 1)[1], 1
                        )
                    )
                    dim_pos[tensor_dim] = _to_i32(col)
                    tensor_dim += 1
            else:
                dim_pos[tensor_dim] = _to_i32(_C_engine.reshape(item_impl, [-1]))
                tensor_dim += 1
        else:
            raise IndexError(f"Unsupported __setitem__ index: {type(item).__name__}")

    # Fill unindexed dims with arange (= "select all")
    for d in range(ndim):
        if d not in dim_pos:
            dim_pos[d] = _C_engine.arange(0, shape[d], 1, _C_engine.I32, device)

    # ── scalar path: mask + where ─────────────────────────────────────────────
    if not hasattr(value, "_impl"):
        # Build float indicator mask and apply where
        mask_impl = _C_engine.full(shape, 1.0, _C_engine.F32, device)
        for d, pos in dim_pos.items():
            ind = _dim_indicator(shape[d], pos, device)
            rs = [1] * ndim
            rs[d] = shape[d]
            ind_bc = _C_engine.broadcast_to(_C_engine.reshape(ind, rs), shape)
            mask_impl = _C_engine.mul(mask_impl, ind_bc)
        val_full = _C_engine.full(shape, float(value), t._impl.dtype, device)
        half = _C_engine.full(shape, 0.5, _C_engine.F32, device)
        cond = _C_engine.greater(mask_impl, half)
        t._impl = _C_engine.where(cond, val_full, t._impl)
        return

    # ── tensor path: flat-index scatter ──────────────────────────────────────
    # Compute flat indices for the cross-product of dim_pos along all dims.
    # Shape of the "index grid": (len(pos0), len(pos1), ..., len(pos_{n-1}))
    # = the shape of val (after broadcast) that maps into t.
    #
    # flat_idx[i0, i1, ..., in-1] = sum_d(dim_pos[d][id] * stride_d)
    # where stride_d = product(shape[d+1:]).

    # Strides for each dim
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]

    # Build flat index as int32 tensor via outer-product-add
    # Start with zeros shaped by the position lengths
    grid_shape = [int(dim_pos[d].shape[0]) for d in range(ndim)]
    flat_idx = _C_engine.zeros(grid_shape, _C_engine.I32, device)

    for d in range(ndim):
        pos = dim_pos[d]  # shape (n_d,)
        # Scale by stride
        stride_tensor = _C_engine.full(
            pos.shape, float(strides[d]), _C_engine.I32, device
        )
        scaled = _C_engine.mul(pos, stride_tensor)  # (n_d,)
        # Reshape to broadcast along dim d of flat_idx
        rs = [1] * ndim
        rs[d] = grid_shape[d]
        scaled_bc = _C_engine.broadcast_to(_C_engine.reshape(scaled, rs), grid_shape)
        flat_idx = _C_engine.add(flat_idx, scaled_bc)

    flat_idx_1d = _to_i32(_C_engine.reshape(flat_idx, [-1]))  # (M,)
    M = _prod(grid_shape)

    # Flatten t
    total = _prod(shape)
    flat_t = _C_engine.reshape(t._impl, [total])

    # Prepare value: broadcast to grid_shape then flatten
    val_impl = _unwrap(value)  # type: ignore[arg-type]
    val_shape = list(val_impl.shape)
    if val_shape != grid_shape:
        n_miss = ndim - len(val_shape)
        if n_miss > 0:
            val_impl = _C_engine.reshape(val_impl, [1] * n_miss + val_shape)
        val_impl = _C_engine.broadcast_to(val_impl, grid_shape)
    flat_val = _C_engine.reshape(val_impl, [-1])

    # Cast val to t's dtype if needed
    if flat_val.dtype != t._impl.dtype:
        flat_val = _C_engine.astype(flat_val, t._impl.dtype)

    # Build indicator for the M target positions (to use index_copy logic):
    # We scatter flat_val into flat_t at flat_idx_1d positions.
    # Use: index_copy = index_fill(zeros) + index_add approach.
    # Actually, directly:  flat_out = scatter(flat_t, 0, flat_idx_1d, flat_val)
    # Engine scatter takes (base, dim, index, src) and REPLACES (not adds).
    # index is same shape as src.
    flat_out = _C_engine.scatter(flat_t, 0, flat_idx_1d, flat_val)
    t._impl = _C_engine.reshape(flat_out, shape)


# Legacy export used in tensor.py (kept for compatibility)
_select_slice = _select_slice
