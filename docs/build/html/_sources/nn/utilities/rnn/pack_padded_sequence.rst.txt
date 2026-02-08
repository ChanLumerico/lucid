nn.utils.rnn.pack_padded_sequence
=================================

.. autofunction:: lucid.nn.utils.rnn.pack_padded_sequence

Function Signature
------------------

.. code-block:: python

    def pack_padded_sequence(
        input_: Tensor,
        lengths: Sequence[int] | Tensor,
        batch_first: bool = False,
        enforce_sorted: bool = True,
    ) -> PackedSequence

Parameters
----------
- **input_** (*Tensor*):
  A padded tensor of shape `(seq_len, batch, feature)` or
  `(batch, seq_len, feature)` when `batch_first=True`.

- **lengths** (*Sequence[int] | Tensor*):
  Sequence lengths for each batch element. Must be 1D and match the batch size.

- **batch_first** (*bool*, optional):
  If True, `input_` is interpreted as `(batch, seq_len, feature)`.
  Default is False.

- **enforce_sorted** (*bool*, optional):
  If True, `lengths` must be sorted in decreasing order. If False, the function
  will sort internally and populate `sorted_indices`/`unsorted_indices`.
  Default is True.

Return Value
------------
- **PackedSequence**:
  A packed representation containing `data`, `batch_sizes`, and optional
  sort/unsort indices.

Behavior
--------
- The maximum sequence length is inferred from `lengths`.
- The returned `batch_sizes` is a 1D tensor of length `max_len`.
- When `enforce_sorted=False`, the input is sorted by length internally.

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    x = lucid.randn(5, 3, 4)  # (seq_len=5, batch=3, feature=4)
    lengths = [5, 3, 2]

    packed = nn.utils.rnn.pack_padded_sequence(x, lengths)
    out, h_n = nn.RNN(4, 6)(packed)
