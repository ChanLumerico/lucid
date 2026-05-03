"""
RNN sequence packing utilities.
"""

from typing import Any, NamedTuple, TYPE_CHECKING
import numpy as np
from lucid._tensor.tensor import Tensor as _Tensor
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._factories.creation import zeros as _zeros
from lucid._factories.converters import tensor as _tensor_fn
from lucid._ops import cat

if TYPE_CHECKING:
    pass


class PackedSequence(NamedTuple):
    """Holds packed padded sequence data."""

    data: Any
    batch_sizes: Any
    sorted_indices: Any
    unsorted_indices: Any


def pack_padded_sequence(
    input: Any,
    lengths: Any,
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    """Pack a padded batch of variable-length sequences."""
    if batch_first:
        input = input.permute(1, 0, *range(2, input.ndim))

    T, B = input.shape[0], input.shape[1]
    lengths_np = np.asarray(
        lengths.numpy() if hasattr(lengths, "numpy") else lengths, dtype=np.int64
    )

    sorted_indices_np = np.argsort(-lengths_np, stable=True)
    unsorted_indices_np = np.argsort(sorted_indices_np, stable=True)

    if enforce_sorted and not np.all(np.diff(lengths_np) <= 0):
        raise ValueError(
            "pack_padded_sequence: sequences must be sorted by length in descending order "
            "or pass enforce_sorted=False."
        )

    if not enforce_sorted:
        input_np = input.numpy()
        input_np = input_np[:, sorted_indices_np]
        lengths_np = lengths_np[sorted_indices_np]
        input = _Tensor(input_np)

    packed_list = []
    batch_sizes_list = []
    for t in range(T):
        bs = int(np.sum(lengths_np > t))
        if bs == 0:
            break
        packed_list.append(input[t, :bs])
        batch_sizes_list.append(bs)

    data_tensor = cat(packed_list, 0)
    bs_arr = np.array(batch_sizes_list, dtype=np.int64)
    batch_sizes_tensor = _wrap(
        _C_engine.TensorImpl(bs_arr, _C_engine.Device.CPU, False)
    )

    si_impl = _C_engine.TensorImpl(
        sorted_indices_np.astype(np.int64), _C_engine.Device.CPU, False
    )
    ui_impl = _C_engine.TensorImpl(
        unsorted_indices_np.astype(np.int64), _C_engine.Device.CPU, False
    )
    return PackedSequence(
        data_tensor, batch_sizes_tensor, _wrap(si_impl), _wrap(ui_impl)
    )


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
) -> tuple[Any, Any]:
    """Unpack a PackedSequence to a padded tensor."""
    data = sequence.data
    batch_sizes_np = np.asarray(sequence.batch_sizes.numpy())

    T = len(batch_sizes_np)
    B = int(batch_sizes_np[0])
    feat_shape = data.shape[1:]

    if total_length is not None and total_length > T:
        T = total_length

    out_np = np.full((T, B) + feat_shape, padding_value, dtype=np.float32)
    offset = 0
    for t, bs in enumerate(batch_sizes_np):
        bs = int(bs)
        out_np[t, :bs] = np.asarray(data[offset : offset + bs].numpy())
        offset += bs

    lengths_np = np.zeros(B, dtype=np.int64)
    for t in range(len(batch_sizes_np) - 1, -1, -1):
        bs = int(batch_sizes_np[t])
        lengths_np[:bs] = np.maximum(lengths_np[:bs], t + 1)

    output = _Tensor(out_np)
    if sequence.unsorted_indices is not None:
        ui = np.asarray(sequence.unsorted_indices.numpy()).astype(np.int64)
        out_np2 = out_np[:, ui]
        lengths_np = lengths_np[ui]
        output = _Tensor(out_np2)

    if batch_first:
        output = output.permute(1, 0, *range(2, output.ndim))

    len_impl = _C_engine.TensorImpl(lengths_np, _C_engine.Device.CPU, False)
    return output, _wrap(len_impl)


def pad_sequence(
    sequences: list[Any],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Any:
    """Pad a list of variable-length tensors."""
    T_max = max(s.shape[0] for s in sequences)
    B = len(sequences)
    feat_shape = sequences[0].shape[1:]

    out_np = np.full((T_max, B) + feat_shape, padding_value, dtype=np.float32)
    for i, s in enumerate(sequences):
        out_np[: s.shape[0], i] = np.asarray(s.numpy())

    if batch_first:
        out_np = np.transpose(out_np, (1, 0) + tuple(range(2, out_np.ndim)))

    return _Tensor(out_np)
