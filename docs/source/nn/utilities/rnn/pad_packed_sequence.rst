nn.utils.rnn.pad_packed_sequence
================================

.. autofunction:: lucid.nn.utils.rnn.pad_packed_sequence

Function Signature
------------------

.. code-block:: python

    def pad_packed_sequence(
        sequence: PackedSequence,
        batch_first: bool = False,
        padding_value: _Scalar = 0,
    ) -> tuple[Tensor, Tensor]

Parameters
----------
- **sequence** (*PackedSequence*):
  Packed input produced by `pack_padded_sequence` or `pack_sequence`.

- **batch_first** (*bool*, optional):
  If True, return padded output as `(batch, seq_len, feature)`.
  Default is False.

- **padding_value** (*_Scalar*, optional):
  Value used to pad shorter sequences. Default is `0`.

Return Value
------------
- **(Tensor, Tensor)**:
  The padded tensor and the lengths tensor of shape `(batch,)`.

Behavior
--------
- Restores padded output and, if `unsorted_indices` exists, returns the batch
  in the original (unsorted) order.

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    x = lucid.randn(4, 2, 3)
    lengths = [4, 2]
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths)

    padded, out_lengths = nn.utils.rnn.pad_packed_sequence(packed)
