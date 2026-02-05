nn.utils.rnn.PackedSequence
===========================

.. autoclass:: lucid.nn.utils.rnn.PackedSequence

`PackedSequence` is a lightweight container that represents a variable-length
batch of sequences in a compact, time-major form. It is produced by
`pack_padded_sequence` or `pack_sequence`, and can be converted back to a padded
tensor with `pad_packed_sequence`.

Class Signature
---------------
.. code-block:: python

    @dataclass(frozen=True)
    class PackedSequence:
        data: Tensor
        batch_sizes: Tensor
        sorted_indices: Tensor | None = None
        unsorted_indices: Tensor | None = None

Attributes
----------
- **data** (*Tensor*):
  Concatenated time steps with shape `(sum(batch_sizes), feature)`.

- **batch_sizes** (*Tensor*):
  1D tensor containing the batch size at each time step. Must be non-increasing.

- **sorted_indices** (*Tensor | None*):
  Indices that sort the original batch by length when `enforce_sorted=False`.

- **unsorted_indices** (*Tensor | None*):
  Inverse permutation for restoring original batch order.

Notes
-----
- `RNN`, `LSTM`, and `GRU` accept a `PackedSequence` input and return a
  `PackedSequence` output with the same `batch_sizes` and index metadata.
- When using packed inputs, `batch_first` has no effect.

Examples
--------
**Packing a padded batch and running an RNN:**

.. code-block:: python

    import lucid
    import lucid.nn as nn

    x = lucid.randn(5, 3, 4)     # (seq_len=5, batch=3, feature=4)
    lengths = [5, 3, 2]
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=True)

    rnn = nn.RNN(input_size=4, hidden_size=6)
    out_packed, h_n = rnn(packed)
    print(out_packed.data.shape)  # (sum(batch_sizes), hidden_size)
    print(h_n.shape)              # (num_layers, batch, hidden_size)

**Packing unsorted sequences and restoring original order:**

.. code-block:: python

    a = lucid.randn(2, 5)
    b = lucid.randn(4, 5)
    c = lucid.randn(1, 5)

    packed = nn.utils.rnn.pack_sequence([a, b, c], enforce_sorted=False)
    padded, out_lengths = nn.utils.rnn.pad_packed_sequence(packed)
    # padded is back in original [a, b, c] order
