from dataclasses import dataclass
from typing import Iterable, Sequence

import lucid

from lucid._tensor import Tensor
from lucid.types import _Scalar


__all__ = [
    "PackedSequence",
    "pad_sequence",
    "pack_padded_sequence",
    "pad_packed_sequence",
    "pack_sequence",
    "unpack_sequence",
]


@dataclass(frozen=True)
class PackedSequence:
    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Tensor | None = None
    unsorted_indices: Tensor | None = None


def pad_sequence(
    sequences: Iterable[Tensor], batch_first: bool = False, padding_value: _Scalar = 0
) -> Tensor:
    seq_list = list(sequences)
    if not seq_list:
        raise ValueError("pad_sequence expected a non-empty iterable of Tensors")

    first = seq_list[0]
    if not isinstance(first, Tensor):
        raise TypeError("pad_sequence expects Tensor elements")

    ndim = first.ndim
    if ndim < 1:
        raise ValueError("pad_sequence expects tensors with at least 1 dimension")

    trailing_shape = first.shape[1:]
    device = first.device
    dtype = first.dtype

    lengths: list[int] = []
    for idx, seq in enumerate(seq_list):
        if not isinstance(seq, Tensor):
            raise TypeError("pad_sequence expects Tensor elements")
        if seq.ndim != ndim:
            raise ValueError(
                f"pad_sequence expects tensors with {ndim} dimensions, "
                f"got {seq.ndim} at index {idx}"
            )
        if seq.shape[1:] != trailing_shape:
            raise ValueError(
                "pad_sequence expects all tensors to share the same trailing shape"
            )
        if seq.device != device:
            raise ValueError("pad_sequence expects all tensors on the same device")
        if seq.dtype != dtype:
            raise ValueError("pad_sequence expects all tensors with the same dtype")
        lengths.append(seq.shape[0])

    max_len = max(lengths)
    batch_size = len(seq_list)

    if batch_first:
        out_shape = (batch_size, max_len, *trailing_shape)
    else:
        out_shape = (max_len, batch_size, *trailing_shape)

    output = lucid.full(out_shape, padding_value, dtype=dtype, device=device)
    for i, seq in enumerate(seq_list):
        length = lengths[i]
        if length == 0:
            continue
        if batch_first:
            output[i, :length] = seq
        else:
            output[:length, i] = seq

    return output


def _as_lengths(lengths: Sequence[int] | Tensor, *, device: str) -> Tensor:
    if isinstance(lengths, Tensor):
        return lengths
    return Tensor(list(lengths), device=device)


def _invert_permutation(indices: Tensor) -> Tensor:
    return lucid.argsort(indices, axis=0)


def pack_padded_sequence(
    input_: Tensor,
    lengths: Sequence[int] | Tensor,
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    if input_.ndim < 2:
        raise ValueError(
            f"pack_padded_sequence expected input with at least 2 dims, got {input_.ndim}"
        )

    if batch_first:
        input_ = input_.swapaxes(0, 1)

    seq_len, batch_size = input_.shape[0], input_.shape[1]
    lengths_t = _as_lengths(lengths, device=input_.device)
    if lengths_t.ndim != 1:
        raise ValueError("lengths must be a 1D sequence or tensor")
    if lengths_t.shape[0] != batch_size:
        raise ValueError(
            f"lengths size {lengths_t.shape[0]} does not match batch size {batch_size}"
        )

    sorted_indices = None
    unsorted_indices = None

    if enforce_sorted:
        sorted_lengths = lengths_t
    else:
        sorted_indices = lucid.argsort(lengths_t, descending=True, axis=0)
        unsorted_indices = _invert_permutation(sorted_indices)

        lengths_t = lengths_t[sorted_indices]
        input_ = input_[:, sorted_indices]
        sorted_lengths = lengths_t

    max_len = int(sorted_lengths[0].item())
    if max_len > seq_len:
        raise ValueError(
            f"lengths has max {max_len} but input has sequence length {seq_len}"
        )

    batch_sizes: list[int] = []
    chunks: list[Tensor] = []
    for t in range(max_len):
        bs = int((sorted_lengths > t).sum().item())
        batch_sizes.append(bs)
        if bs == 0:
            break
        chunks.append(input_[t, :bs])

    if not chunks:
        data = input_[:0]
    else:
        data = lucid.concatenate(tuple(chunks), axis=0)

    return PackedSequence(
        data=data,
        batch_sizes=Tensor(batch_sizes, device=input_.device),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def pad_packed_sequence(
    sequence: PackedSequence, batch_first: bool = False, padding_value: _Scalar = 0
) -> tuple[Tensor, Tensor]:
    data = sequence.data
    batch_sizes = sequence.batch_sizes
    if batch_sizes.ndim != 1:
        raise ValueError("batch_sizes must be 1D")

    max_len = int(batch_sizes.shape[0])
    if max_len == 0:
        raise ValueError("batch_sizes must be non-empty")

    batch_size = int(batch_sizes[0].item())
    trailing_shape = data.shape[1:]

    if batch_first:
        out_shape = (batch_size, max_len, *trailing_shape)
    else:
        out_shape = (max_len, batch_size, *trailing_shape)

    output = lucid.full(out_shape, padding_value, dtype=data.dtype, device=data.device)

    lengths = [0] * batch_size
    offset = 0
    for t in range(max_len):
        bs = int(batch_sizes[t].item())
        if bs == 0:
            break

        chunk = data[offset : offset + bs]
        offset += bs
        for i in range(bs):
            lengths[i] += 1
        if batch_first:
            output[:bs, t] = chunk
        else:
            output[t, :bs] = chunk

    lengths_t = Tensor(lengths, device=data.device)
    if sequence.unsorted_indices is not None:
        if batch_first:
            output = output[sequence.unsorted_indices]
        else:
            output = output[:, sequence.unsorted_indices]
        lengths_t = lengths_t[sequence.unsorted_indices]

    return output, lengths_t


def pack_sequence(
    sequences: Iterable[Tensor], enforce_sorted: bool = True
) -> PackedSequence:
    seq_list = list(sequences)
    if not seq_list:
        raise ValueError("pack_sequence expected a non-empty iterable of Tensors")

    lengths = [seq.shape[0] for seq in seq_list]
    padded = pad_sequence(seq_list, batch_first=False, padding_value=0.0)
    return pack_padded_sequence(
        padded, lengths, batch_first=False, enforce_sorted=enforce_sorted
    )


def unpack_sequence(
    sequence: PackedSequence, batch_first: bool = False
) -> list[Tensor]:
    padded, lengths = pad_packed_sequence(
        sequence, batch_first=batch_first, padding_value=0.0
    )
    result: list[Tensor] = []
    for i, length in enumerate(lengths):
        l = int(length.item())
        if batch_first:
            result.append(padded[i, :l])
        else:
            result.append(padded[:l, i])
    return result
