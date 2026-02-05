nn.LSTM
=======

.. autoclass:: lucid.nn.LSTM

`LSTM` is the user-facing long short-term memory layer that wraps `RNNBase` in
`"LSTM"` mode. It processes full sequences and returns the output sequence along
with the final hidden and cell states for each stacked layer.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.LSTM(
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    )

Parameters
----------
- **input_size** (*int*):
  Number of expected features in the input at each time step.

- **hidden_size** (*int*):
  Number of features in the hidden and cell state for every layer.

- **num_layers** (*int*, optional):
  How many stacked LSTM layers to run. Default: `1`.

- **bias** (*bool*, optional):
  If `True`, each layer uses input and hidden biases. Default: `True`.

- **batch_first** (*bool*, optional):
  If `True`, inputs/outputs use `(batch, seq_len, feature)` layout; otherwise
  `(seq_len, batch, feature)`. Default: `False`.

- **dropout** (*float*, optional):
  Dropout probability applied to outputs of all layers except the last when
  `self.training` is `True`. Default: `0.0`.

Inputs and Outputs
------------------
- **Input**: `(seq_len, batch, input_size)` or `(batch, seq_len, input_size)` when
  `batch_first=True`.
- **Packed input**: `PackedSequence` with `data` shaped
  `(sum(batch_sizes), input_size)`. When packed, `batch_first` has no effect.
- **Initial state `(h_0, c_0)`**: each shaped `(num_layers, batch, hidden_size)`. If
  omitted, zero-initialized states are created. 2D tensors are accepted and expanded
  to the first layer.
- **Returns**: `(output, (h_n, c_n))`

  - `output`: same leading dimensions as the input, with feature size `hidden_size`.
  - `output` is a `PackedSequence` when the input is packed.
  - `h_n` and `c_n`: final hidden and cell states for each layer, each shaped
    `(num_layers, batch, hidden_size)`.

Examples
--------
**Single-layer LSTM over a batch-first sequence:**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> seq = lucid.randn(3, 5, 4)  # (batch=3, seq_len=5, input_size=4)
    >>> lstm = nn.LSTM(input_size=4, hidden_size=6, batch_first=True)
    >>> output, (h_n, c_n) = lstm(seq)
    >>> output.shape, h_n.shape, c_n.shape
    ((3, 5, 6), (1, 3, 6), (1, 3, 6))

**Stacked LSTM with dropout and custom initial state:**

.. code-block:: python

    >>> h0 = lucid.zeros(2, 2, 8)
    >>> c0 = lucid.zeros(2, 2, 8)
    >>> seq = lucid.randn(7, 2, 10)
    >>> lstm = nn.LSTM(
    ...     input_size=10,
    ...     hidden_size=8,
    ...     num_layers=2,
    ...     dropout=0.1,
    ... )
    >>> output, (h_n, c_n) = lstm(seq, (h0, c0))
    >>> output.shape
    (7, 2, 8)
