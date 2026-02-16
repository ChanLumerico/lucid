nn.utils.rnn.pad_sequence
=========================

.. autofunction:: lucid.nn.utils.rnn.pad_sequence

Function Signature
------------------

.. code-block:: python

    def pad_sequence(
        sequences: Iterable[Tensor],
        batch_first: bool = False,
        padding_value: _Scalar = 0.0,
    ) -> Tensor

Parameters
----------

- **sequences** (*Iterable[Tensor]*):
  A list (or iterable) of sequence tensors. Each tensor must share the same
  trailing shape (all dimensions except the first), and all tensors must be on
  the same device with the same dtype. The first dimension is treated as the
  sequence length.

- **batch_first** (*bool*, optional):
  If True, output shape is `(batch, max_len, *trailing_shape)`. If False,
  output shape is `(max_len, batch, *trailing_shape)`. Default is False.

- **padding_value** (*_Scalar*, optional):
  Value used for padding shorter sequences. Default is `0.0`.

Return Value
------------

- **Tensor**:
  A padded tensor containing all sequences stacked along a new batch axis.

Behavior
--------

- The maximum sequence length is computed from the input list.
- Sequences shorter than `max_len` are padded with `padding_value`.
- This function expects a non-empty iterable of tensors.

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    a = lucid.Tensor([[1.0, 2.0], [3.0, 4.0]])      # length 2
    b = lucid.Tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])  # length 3

    out = nn.utils.rnn.pad_sequence([a, b], batch_first=False, padding_value=0.0)
    print(out.shape)  # (3, 2, 2)

Usage Tips
----------

.. tip::

   Use `batch_first=True` if your RNN modules expect input shaped as
   `(batch, seq, feature)`.
