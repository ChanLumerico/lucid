nn.GRU
======

.. autoclass:: lucid.nn.GRU

`GRU` is the gated recurrent unit layer that wraps `RNNBase` in `"GRU"` mode. It
processes sequences using reset and update gates and returns both the full output
sequence and the final hidden state for each stacked layer.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.GRU(
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
  Number of features in the hidden state for every layer.

- **num_layers** (*int*, optional):
  How many stacked GRU layers to run. Default: `1`.

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
- **Initial hidden state `hx`**: `(num_layers, batch, hidden_size)`. If omitted, a
  zero tensor is created. A 2D `(batch, hidden_size)` tensor is accepted and expanded
  to the first layer.
- **Returns**: `(output, h_n)`

  - `output`: same leading dimensions as the input, with feature size `hidden_size`.
  - `output` is a `PackedSequence` when the input is packed.
  - `h_n`: final hidden state for each layer with shape `(num_layers, batch, hidden_size)`.

Examples
--------
**Basic GRU over a sequence-first input:**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> seq = lucid.randn(6, 4, 3)  # (seq_len=6, batch=4, input_size=3)
    >>> gru = nn.GRU(input_size=3, hidden_size=5)
    >>> output, h_n = gru(seq)
    >>> output.shape, h_n.shape
    ((6, 4, 5), (1, 4, 5))

**Stacked GRU with dropout and explicit initial state:**

.. code-block:: python

    >>> h0 = lucid.zeros(2, 3, 7)
    >>> seq = lucid.randn(5, 3, 4)
    >>> gru = nn.GRU(
    ...     input_size=4,
    ...     hidden_size=7,
    ...     num_layers=2,
    ...     dropout=0.2,
    ... )
    >>> output, h_n = gru(seq, h0)
    >>> (output.shape, h_n.shape)
    ((5, 3, 7), (2, 3, 7))
