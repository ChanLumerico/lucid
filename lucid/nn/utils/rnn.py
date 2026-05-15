"""
RNN sequence packing utilities.
All computation uses the C++ engine — no numpy.
"""

import struct
from typing import NamedTuple, cast

import lucid
from lucid._tensor.tensor import Tensor as _Tensor
from lucid._tensor.tensor import Tensor  # public alias for NamedTuple field annotations
from lucid._C import engine as _C_engine
from lucid._ops import cat  # type: ignore[attr-defined]  # cat is in _ops.__init__ but not re-exported with __all__


# Integer-dtype → struct format-code map, used by _int_tensor_to_list to
# unpack a 1-D int tensor's raw bytes into a Python list without going
# through numpy.  Apple Silicon is little-endian; ``=`` uses native byte
# order without struct padding, which matches the engine's contiguous
# bytes layout exactly.
_INT_DTYPE_STRUCT: dict[_C_engine.Dtype, str] = {
    _C_engine.Dtype.I8: "b",
    _C_engine.Dtype.I16: "h",
    _C_engine.Dtype.I32: "i",
    _C_engine.Dtype.I64: "q",
    _C_engine.Dtype.Bool: "?",
}


def _int_tensor_to_list(t: Tensor) -> list[int]:
    """Convert a 1-D integer tensor to a flat Python list of ints — numpy-free.

    Reads the tensor's contiguous bytes through ``to_bytes`` (which exists
    precisely to keep paths off the numpy bridge) and unpacks them with
    ``struct.unpack``.  Used by ``pack_padded_sequence`` /
    ``pad_packed_sequence`` to inspect ``lengths`` / ``batch_sizes`` /
    ``unsorted_indices`` — small metadata tensors where numpy was vastly
    overkill.

    Raises ``TypeError`` for non-integer dtypes (we intentionally don't
    silently coerce floats — the caller should pass an int tensor).
    """
    n = t._impl.numel()
    if n == 0:
        return []
    fmt = _INT_DTYPE_STRUCT.get(t._impl.dtype)
    if fmt is None:
        raise TypeError(
            f"_int_tensor_to_list expects an integer tensor; got dtype {t._impl.dtype!r}"
        )
    raw = t._impl.to_bytes()
    return list(struct.unpack(f"={n}{fmt}", raw))


class PackedSequence(NamedTuple):
    r"""Compact representation of a batch of variable-length sequences.

    A ``PackedSequence`` interleaves all surviving time-steps of every
    sequence in the batch into a single flat tensor, dropping the
    padding entries entirely.  RNN cells consume this form to avoid
    wasting compute on padded positions.

    Attributes
    ----------
    data : Tensor
        Concatenation of the active features at each time-step in
        descending-length order, shape ``(sum(batch_sizes), *feat)``.
    batch_sizes : Tensor
        1-D int tensor giving the number of sequences still alive at
        each successive time-step.  Strictly non-increasing.
    sorted_indices : Tensor or None
        Permutation that took the original batch order to the packed
        (descending-length) order; ``None`` if the input was already
        sorted.
    unsorted_indices : Tensor or None
        Inverse of ``sorted_indices``; used by
        :func:`pad_packed_sequence` to restore the caller's original
        batch order on unpack.

    Notes
    -----
    Conceptually, with sequence lengths :math:`\ell_1 \geq \dots \geq
    \ell_B`, the packed layout walks time-major:

    .. math::

        \text{data} = \bigl[\,x^{(1)}_0, \dots, x^{(B)}_0,\;
                            x^{(1)}_1, \dots, x^{(B_1)}_1,\;
                            \dots\,\bigr],

    where :math:`B_t` (= ``batch_sizes[t]``) is the count of sequences
    with length :math:`> t`.

    Examples
    --------
    >>> from lucid.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    >>> # padded (B=3, T=4, F=5), sorted-by-length [4, 3, 2]
    >>> packed = pack_padded_sequence(x_padded, lengths=[4, 3, 2], batch_first=True)
    >>> # ... feed to RNN ...
    >>> unpacked, lengths = pad_packed_sequence(packed, batch_first=True)
    """

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
    r"""Pack a padded :math:`(T, B, *)` batch into a :class:`PackedSequence`.

    Strips out the padding cells so downstream RNN kernels iterate only
    over genuine time-steps.  Reduces both compute and (with masked
    losses) accidental gradient flow through pad positions.

    Parameters
    ----------
    input : Tensor
        Padded batch.  Default layout is ``(T, B, *)`` with time on
        axis 0; set ``batch_first=True`` for ``(B, T, *)``.
    lengths : Tensor or list of int
        Per-sequence true lengths.  Shape ``(B,)``.  When passed as a
        tensor it must be a 1-D integer tensor.
    batch_first : bool, optional
        Whether ``input`` is laid out batch-first.  Default ``False``.
    enforce_sorted : bool, optional
        If ``True`` (the default), assume the caller already supplied
        sequences in descending-length order — cheap but raises
        :class:`ValueError` on violation.  Set to ``False`` to have the
        function sort internally; the sort permutation is stored on the
        returned :class:`PackedSequence` so :func:`pad_packed_sequence`
        can undo it.

    Returns
    -------
    PackedSequence
        Packed view of the batch.

    Raises
    ------
    ValueError
        With ``enforce_sorted=True`` and an unsorted ``lengths``.

    Notes
    -----
    Total packed length equals :math:`\sum_b \ell_b`, the sum of true
    sequence lengths — strictly less than :math:`T \cdot B` whenever any
    sequence is shorter than the max.

    Examples
    --------
    >>> from lucid.nn.utils.rnn import pack_padded_sequence
    >>> packed = pack_padded_sequence(x, lengths=[5, 3, 2], batch_first=False)
    """
    if batch_first:
        input = input.permute(1, 0, *range(2, input.ndim))

    T, B = int(input.shape[0]), int(input.shape[1])

    # Extract lengths as a plain Python list of ints (small metadata, not tensor math).
    if hasattr(lengths, "_impl"):
        # Lucid Tensor: read raw bytes + struct.unpack (numpy-free path).
        lengths_list: list[int] = _int_tensor_to_list(cast(Tensor, lengths))
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
        _sorted_t = _Tensor(sorted_idx)  # type: ignore[arg-type]  # list[int] is list[object] at runtime
        si_impl_i32 = _C_engine.astype(_sorted_t._impl, _C_engine.I32)
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
    bs_t = _Tensor(batch_sizes_list)  # type: ignore[arg-type]  # list[int] is list[object] at runtime
    si_t = _Tensor(sorted_idx)  # type: ignore[arg-type]  # list[int] is list[object] at runtime
    ui_t = _Tensor(unsorted_idx)  # type: ignore[arg-type]  # list[int] is list[object] at runtime

    return PackedSequence(data_tensor, bs_t, si_t, ui_t)


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
) -> tuple[Any, Any]:
    r"""Inverse of :func:`pack_padded_sequence` — produce a padded tensor.

    Re-inflates a :class:`PackedSequence` into a dense
    :math:`(T_\text{max}, B, *)` (or :math:`(B, T_\text{max}, *)`)
    tensor with the original batch order restored.  The freshly created
    cells beyond each sequence's true length are filled with
    ``padding_value``.

    Parameters
    ----------
    sequence : PackedSequence
        Packed batch to unpack.
    batch_first : bool, optional
        Output layout.  ``False`` (default) yields ``(T, B, *)``,
        ``True`` yields ``(B, T, *)``.
    padding_value : float, optional
        Fill value for padded entries.  Default ``0.0``.
    total_length : int, optional
        If given, pad up to this length instead of the longest packed
        sequence.  Useful when downstream code expects a fixed
        ``T_max`` (e.g. fully-static export graphs).  Must be ``>=`` the
        longest sequence.

    Returns
    -------
    Tensor
        Padded batch tensor.
    Tensor
        1-D tensor of per-sequence true lengths in the *original* batch
        order.

    Notes
    -----
    Output time dimension is :math:`\max_b \ell_b` (or ``total_length``
    if larger).  The function honours ``unsorted_indices`` on the
    packed sequence so the rows of the returned tensor match the order
    the caller used at pack time.

    Examples
    --------
    >>> from lucid.nn.utils.rnn import pad_packed_sequence
    >>> padded, lens = pad_packed_sequence(packed, batch_first=True)
    """
    data = sequence.data
    # batch_sizes is a small 1-D int tensor — extract as Python list (metadata).
    # Uses to_bytes + struct.unpack so this code path stays numpy-free.
    batch_sizes_list = _int_tensor_to_list(sequence.batch_sizes)

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
        # Reorder batch dim by unsorted_indices (numpy-free metadata read).
        ui_list = _int_tensor_to_list(sequence.unsorted_indices)
        ui_t = _Tensor(cast(list[object], ui_list))
        ui_impl_i32 = _C_engine.astype(ui_t._impl, _C_engine.I32)
        full_shape = list(out_t.shape)
        bcast_shape = [1, B] + [1] * (out_t.ndim - 2)
        idx_rs = _C_engine.reshape(ui_impl_i32, bcast_shape)
        idx_bc = _C_engine.broadcast_to(idx_rs, full_shape)
        out_t = _Tensor.__new_from_impl__(_C_engine.gather(out_t._impl, idx_bc, 1))
        lengths_list = [lengths_list[ui_list[i]] for i in range(B)]

    if batch_first:
        out_t = out_t.permute(1, 0, *range(2, out_t.ndim))

    len_t = _Tensor(cast(list[object], lengths_list))
    return out_t, len_t


def pad_sequence(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    r"""Stack a list of variable-length tensors into a padded batch.

    Each input is padded along its leading time dimension up to the
    longest sequence in ``sequences``, then the padded versions are
    stacked along a new batch axis.  Useful as the first step of an
    RNN training loop when sequences are produced one at a time (per-
    example tokenisation, audio framing, etc.).

    Parameters
    ----------
    sequences : list of Tensor
        Non-empty list of per-example tensors.  Each must share the
        same trailing feature shape and dtype / device; only the
        leading length axis may differ.
    batch_first : bool, optional
        Output layout.  ``False`` (default) yields ``(T_max, B, *feat)``,
        ``True`` yields ``(B, T_max, *feat)``.
    padding_value : float, optional
        Fill value for padded entries.  Default ``0.0``.

    Returns
    -------
    Tensor
        Padded batch tensor.

    Raises
    ------
    ValueError
        If ``sequences`` is empty.

    Notes
    -----
    Padding is constructed by concatenating a freshly allocated tail
    of shape ``(T_max - T_i, *feat)`` onto each input — this keeps the
    operation differentiable end-to-end (view-based mutation would
    silently drop gradients through Lucid's autograd-aware
    :class:`~lucid._tensor.tensor.Tensor`).

    Examples
    --------
    >>> from lucid.nn.utils.rnn import pad_sequence
    >>> batch = pad_sequence([a, b, c], batch_first=True)
    """
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
) -> PackedSequence:
    r"""One-shot pack-from-list — equivalent to :func:`pad_sequence` then :func:`pack_padded_sequence`.

    Convenience wrapper for the common case where a list of per-example
    tensors needs to become a :class:`PackedSequence` without going
    through an intermediate padded tensor.  Each entry's leading axis
    is treated as the time dimension; remaining axes carry the feature
    shape (which must be identical across the list).

    Parameters
    ----------
    sequences : list of Tensor
        Per-example tensors.  Leading axis is time; trailing axes are
        the feature shape and must match across entries.
    enforce_sorted : bool, optional
        If ``True`` (the default), the caller must supply sequences in
        non-increasing length order; mismatches raise
        :class:`ValueError`.  ``False`` is *not* currently supported
        and raises :class:`NotImplementedError` — sort externally for
        now.

    Returns
    -------
    PackedSequence
        Packed view of the input list.

    Raises
    ------
    ValueError
        If ``sequences`` is empty, or if it is not sorted while
        ``enforce_sorted=True``.
    NotImplementedError
        If ``enforce_sorted=False`` is requested.

    Notes
    -----
    Implemented as :func:`pad_sequence` followed by
    :func:`pack_padded_sequence`.  The intermediate padded tensor lives
    only for the duration of the call.

    Examples
    --------
    >>> from lucid.nn.utils.rnn import pack_sequence
    >>> packed = pack_sequence([long_seq, mid_seq, short_seq])
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
