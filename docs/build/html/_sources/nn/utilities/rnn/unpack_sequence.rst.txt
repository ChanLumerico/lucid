nn.utils.rnn.unpack_sequence
============================

.. autofunction:: lucid.nn.utils.rnn.unpack_sequence

Function Signature
------------------

.. code-block:: python

    def unpack_sequence(
        sequence: PackedSequence,
        batch_first: bool = False,
    ) -> list[Tensor]

Parameters
----------
- **sequence** (*PackedSequence*):
  Packed input produced by `pack_padded_sequence` or `pack_sequence`.

- **batch_first** (*bool*, optional):
  If True, each returned tensor has shape `(seq_len, feature)` with batch-first
  padded output internally. Default is False.

Return Value
------------
- **list[Tensor]**:
  List of individual sequences (unpadded), restored to original ordering when
  available.

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    a = lucid.randn(2, 4)
    b = lucid.randn(3, 4)
    packed = nn.utils.rnn.pack_sequence([a, b], enforce_sorted=False)

    seqs = nn.utils.rnn.unpack_sequence(packed)
    print([s.shape for s in seqs])
