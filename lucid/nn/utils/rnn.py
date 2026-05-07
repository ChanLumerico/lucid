"""
RNN sequence packing utilities.
All computation uses the C++ engine — no numpy.
"""

from typing import Any, NamedTuple, TYPE_CHECKING
from lucid._tensor.tensor import Tensor as _Tensor
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._ops import cat


class PackedSequence(NamedTuple):
    """Holds packed padded sequence data."""

    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Tensor | None
    unsorted_indices: Tensor | None


def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor | list[int],
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    """Pack a padded batch of variable-length sequences."""
    if batch_first:
        input = input.permute(1, 0, *range(2, input.ndim))

    T, B = int(input.shape[0]), int(input.shape[1])

    # Extract lengths as a plain Python list of ints (small metadata, not tensor math).
    if hasattr(lengths, "_impl"):
        # Lucid Tensor: download via data_as_python (interop boundary — no computation)
        raw = lengths._impl.data_as_python()
        lengths_list: list[int] = [int(raw.flat[i]) for i in range(len(raw.flat))]
    else:
        lengths_list = [int(v) for v in lengths]

    # Sort by descending length with Python builtins — no numpy argsort.
    sorted_idx = sorted(range(B), key=lambda i: -lengths_list[i])
    unsorted_idx = [0] * B
    for new_pos, orig_pos in enumerate(sorted_idx):
        unsorted_idx[orig_pos] = new_pos

    if enforce_sorted:
        for i in range(len(lengths_list) - 1):
            if lengths_list[i] < lengths_list[i + 1]:
                raise ValueError(
                    "pack_padded_sequence: sequences must be sorted by length in "
                    "descending order or pass enforce_sorted=False."
                )

    # Reorder input along batch dim using engine gather.
    if not enforce_sorted:
        # Build sorted-index TensorImpl from Python list (interop boundary — metadata only).
        si_impl_i32 = _C_engine.astype(_Tensor(sorted_idx)._impl, _C_engine.I32)
        bcast_shape = [1, B] + [1] * (input.ndim - 2)
        full_shape = list(input.shape)
        idx_rs = _C_engine.reshape(si_impl_i32, bcast_shape)
        idx_bc = _C_engine.broadcast_to(idx_rs, full_shape)
        input = _Tensor.__new_from_impl__(_C_engine.gather(input._impl, idx_bc, 1))
        lengths_list = [lengths_list[i] for i in sorted_idx]

    # Pack: collect active elements per timestep.
    packed_list = []
    batch_sizes_list = []
    for t in range(T):
        bs = sum(1 for l in lengths_list if l > t)
        if bs == 0:
            break
        packed_list.append(input[t, :bs])
        batch_sizes_list.append(bs)

    data_tensor = cat(packed_list, 0)

    # Build int tensors for batch_sizes, sorted_indices, unsorted_indices.
    # These are metadata (tiny 1-D integer arrays) — construct via Tensor([...]) interop.
    bs_t = _Tensor(batch_sizes_list)
    si_t = _Tensor(sorted_idx)
    ui_t = _Tensor(unsorted_idx)

    return PackedSequence(data_tensor, bs_t, si_t, ui_t)


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
) -> tuple[Any, Any]:
    """Unpack a PackedSequence to a padded tensor."""
    data = sequence.data
    # batch_sizes is a small 1-D int tensor — extract as Python list (metadata).
    bs_raw = sequence.batch_sizes._impl.data_as_python()
    batch_sizes_list = [int(bs_raw.flat[i]) for i in range(len(bs_raw.flat))]

    T = len(batch_sizes_list)
    B = int(batch_sizes_list[0])
    feat_shape = list(data.shape[1:])

    if total_length is not None and total_length > T:
        T = total_length

    # Create padded output with engine full, then fill via copy_from per timestep.
    out_shape = [T, B] + feat_shape
    out_impl = _C_engine.full(
        out_shape, padding_value, data._impl.dtype, data._impl.device
    )
    out_t = _Tensor.__new_from_impl__(out_impl)

    offset = 0
    for t, bs in enumerate(batch_sizes_list):
        # data[offset:offset+bs] → place into out_t[t, :bs]
        src = data[offset : offset + bs]  # (bs, *feat)
        # Assign via index (Python-level setitem calls copy_from internally)
        out_t[t, :bs].copy_(src)
        offset += bs

    # Compute lengths from batch_sizes.
    lengths_list = [0] * B
    for t in range(len(batch_sizes_list) - 1, -1, -1):
        bs = batch_sizes_list[t]
        for k in range(bs):
            if lengths_list[k] < t + 1:
                lengths_list[k] = t + 1

    if sequence.unsorted_indices is not None:
        # Reorder batch dim by unsorted_indices.
        ui_raw = sequence.unsorted_indices._impl.data_as_python()
        ui_list = [int(ui_raw.flat[i]) for i in range(len(ui_raw.flat))]
        ui_t = _Tensor(ui_list)
        ui_impl_i32 = _C_engine.astype(ui_t._impl, _C_engine.I32)
        full_shape = list(out_t.shape)
        bcast_shape = [1, B] + [1] * (out_t.ndim - 2)
        idx_rs = _C_engine.reshape(ui_impl_i32, bcast_shape)
        idx_bc = _C_engine.broadcast_to(idx_rs, full_shape)
        out_t = _Tensor.__new_from_impl__(_C_engine.gather(out_t._impl, idx_bc, 1))
        lengths_list = [lengths_list[ui_list[i]] for i in range(B)]

    if batch_first:
        out_t = out_t.permute(1, 0, *range(2, out_t.ndim))

    len_t = _Tensor(lengths_list)
    return out_t, len_t


def pad_sequence(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    """Pad a list of variable-length tensors.

    Each sequence is padded along its leading dimension to the longest
    length in ``sequences``, then stacked along a new batch axis.  The
    older slice-and-copy implementation relied on view-based mutation
    that didn't propagate through Lucid's autograd-aware Tensor — we now
    build the padded outputs explicitly via ``cat`` over the time axis
    and ``stack`` over the batch axis, which goes through proper
    differentiable ops.
    """
    import lucid

    if not sequences:
        raise ValueError("pad_sequence: empty input list")
    T_max: int = max(int(s.shape[0]) for s in sequences)
    feat_shape: list[int] = list(sequences[0].shape[1:])
    dtype = sequences[0].dtype
    device = sequences[0]._impl.device

    padded_each: list[Tensor] = []
    for s in sequences:
        T_i: int = int(s.shape[0])
        if T_i == T_max:
            padded_each.append(s)
            continue
        # Build a constant-fill tail and concat.
        tail_shape: list[int] = [T_max - T_i] + feat_shape
        tail_impl = _C_engine.full(tail_shape, padding_value, s._impl.dtype, device)
        tail: Tensor = _Tensor.__new_from_impl__(tail_impl)
        padded_each.append(lucid.cat([s, tail], 0))

    # Stack along axis 1 (T_max, B, *feat) or axis 0 with batch_first.
    if batch_first:
        return lucid.stack(padded_each, 0)
    return lucid.stack(padded_each, 1)


def pack_sequence(
    sequences: list[Tensor],
    enforce_sorted: bool = True,
) -> "PackedSequence":
    """Pack a list of variable-length sequences into a ``PackedSequence``.

    This is the convenience wrapper around ``pad_sequence`` +
    ``pack_padded_sequence`` that the reference framework exposes — useful
    when you already have a list of per-example tensors and don't want to
    pad them yourself.  Each entry's ``shape[0]`` is treated as the time
    dimension; remaining axes carry the feature dimensions.

    When ``enforce_sorted=True`` (the default) the caller must pass the
    sequences in non-increasing length order; otherwise ``ValueError`` is
    raised.  We don't currently re-sort under the hood, so explicit ordering
    is the only path for now.
    """
    if not sequences:
        raise ValueError("pack_sequence: empty input list")
    lengths: list[int] = [int(s.shape[0]) for s in sequences]
    if enforce_sorted:
        for i in range(1, len(lengths)):
            if lengths[i] > lengths[i - 1]:
                raise ValueError(
                    "pack_sequence: lengths must be sorted in decreasing order "
                    "when enforce_sorted=True"
                )
    elif sorted(lengths, reverse=True) != lengths:
        raise NotImplementedError(
            "pack_sequence: enforce_sorted=False not yet implemented; "
            "sort sequences by descending length before calling"
        )
    padded: Tensor = pad_sequence(sequences, batch_first=False)
    return pack_padded_sequence(padded, lengths, batch_first=False)
