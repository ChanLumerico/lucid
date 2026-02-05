nn.RNN
======

.. autoclass:: lucid.nn.RNN

`RNN` is the user-facing simple recurrent layer that wraps `RNNBase` and selects
the nonlinearity via a friendly argument (`"tanh"` or `"relu"`). It processes
entire sequences and returns both the full output sequence and the final hidden
state for each stacked layer.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.RNN(
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
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
  Number of stacked recurrent layers. Default: `1`.

- **nonlinearity** (*Literal["tanh", "relu"]*, optional):
  Activation applied inside each recurrent cell. Default: `"tanh"`.

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
- **Initial hidden state `hx`**: `(num_layers, batch, hidden_size)`. If omitted,
  zero-initialized hidden states are created. A 2D `(batch, hidden_size)` tensor
  is accepted and expanded to the first layer.
- **Returns**: `(output, h_n)`

  - `output`: same leading dimensions as the input, with feature size `hidden_size`.
  - `output` is a `PackedSequence` when the input is packed.
  - `h_n`: final hidden state for each layer with shape `(num_layers, batch, hidden_size)`.

Examples
--------
**Basic tanh RNN over a batch-first sequence:**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> seq = lucid.randn(4, 6, 10)  # (batch=4, seq_len=6, input_size=10)
    >>> rnn = nn.RNN(input_size=10, hidden_size=12, batch_first=True)
    >>> output, h_n = rnn(seq)
    >>> output.shape
    (4, 6, 12)
    >>> h_n.shape
    (1, 4, 12)

**Stacked ReLU RNN with dropout and custom initial state:**

.. code-block:: python

    >>> seq = lucid.randn(5, 3, 8)  # (seq_len=5, batch=3, input_size=8)
    >>> h0 = lucid.zeros(2, 3, 6)   # (num_layers=2, batch=3, hidden_size=6)
    >>> rnn = nn.RNN(
    ...     input_size=8,
    ...     hidden_size=6,
    ...     num_layers=2,
    ...     nonlinearity="relu",
    ...     dropout=0.2,
    ... )
    >>> output, h_n = rnn(seq, h0)
    >>> (output.shape, h_n.shape)
    ((5, 3, 6), (2, 3, 6))
