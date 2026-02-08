nn.utils.rnn.pack_sequence
==========================

.. autofunction:: lucid.nn.utils.rnn.pack_sequence

Function Signature
------------------

.. code-block:: python

    def pack_sequence(
        sequences: Iterable[Tensor],
        enforce_sorted: bool = True,
    ) -> PackedSequence

Parameters
----------
- **sequences** (*Iterable[Tensor]*):
  Iterable of tensors, each shaped `(seq_len, feature)` with matching trailing
  dimensions.

- **enforce_sorted** (*bool*, optional):
  If True, the sequences must be sorted by length in decreasing order.
  If False, the function will sort internally.

Return Value
------------
- **PackedSequence**:
  Packed representation of the input sequences.

Behavior
--------
- Internally calls `pad_sequence` and `pack_padded_sequence`.

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    a = lucid.randn(3, 5)
    b = lucid.randn(1, 5)

    packed = nn.utils.rnn.pack_sequence([a, b], enforce_sorted=False)
