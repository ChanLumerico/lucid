nn.RNNBase
==========

.. autoclass:: lucid.nn.RNNBase

`RNNBase` implements stacked recurrent layers built from `nn.RNNCell` (simple RNN)
or `nn.LSTMCell` depending on the selected mode. It runs full sequences and returns
per-time-step outputs along with the final hidden state(s) for each layer. Both
sequence-first (`(seq_len, batch, input_size)`) and batch-first
(`(batch, seq_len, input_size)`) layouts are supported, and packed inputs are
accepted via `nn.utils.rnn.PackedSequence`.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.RNNBase(
        mode: Literal["RNN_TANH", "RNN_RELU"],
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
    dropout: float = 0.0,
    )

Parameters
----------
- **mode** (*Literal["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]*):
    Selects the recurrent cell type: simple RNN with `tanh` or `relu`, gated
    `LSTM` using `nn.LSTMCell`, or `GRU` using `nn.GRUCell`.

- **input_size** (*int*):
  Number of expected features in the input at each time step.

- **hidden_size** (*int*):
  Number of features in the hidden state for every layer.

- **num_layers** (*int*, optional):
  How many stacked recurrent layers to run. Default: `1`.

- **bias** (*bool*, optional):
  If `True`, each `RNNCell` uses input and hidden biases. Default: `True`.

- **batch_first** (*bool*, optional):
  If `True`, input and output tensors use shape `(batch, seq_len, feature)`.
  Otherwise `(seq_len, batch, feature)`. Default: `False`.

- **dropout** (*float*, optional):
  Dropout probability applied to outputs of all layers except the last, only
  when `self.training` is `True`. Default: `0.0`.

Attributes
----------
- **layers** (*ModuleList*):
    Sequence of recurrent cell instances, one per layer (`RNNCell`, `LSTMCell`,
    or `GRUCell`).

- **mode** (*str*):
  The internal mode string (`"RNN_TANH"` or `"RNN_RELU"`).

- **nonlinearity** (*str*):
  The activation name (`"tanh"` or `"relu"`) used by the cells.

- **input_size**, **hidden_size**, **num_layers**, **bias**, **batch_first**, **dropout**:
  Constructor arguments stored for reference.

Forward Calculation
-------------------
Given an input sequence :math:`x` and optional initial state, the module computes
per-layer hidden states using its stacked cells:

.. math::

    h_t^{(l)} = \sigma\!\left(x_t^{(l)} W_{ih}^{(l)T} + b_{ih}^{(l)}
        + h_{t-1}^{(l)} W_{hh}^{(l)T} + b_{hh}^{(l)}\right)

Where:

- :math:`x_t^{(0)} = x_t` and :math:`x_t^{(l)} = h_t^{(l-1)}` for :math:`l > 0`.
- :math:`\sigma` is `tanh` when `mode="RNN_TANH"` or `ReLU` when `mode="RNN_RELU"`.
  When `mode="LSTM"`, gating follows the standard LSTM cell equations using
  `i_t, f_t, g_t, o_t` gates inside each `LSTMCell`. For `mode="GRU"`, gating
  follows the standard GRU equations with reset, update, and candidate gates.
- Dropout (if enabled and not the last layer) is applied to :math:`h_t^{(l)}` before
  it is fed into the next layer.

Input and Output Shapes
-----------------------
- **Input**: `(seq_len, batch, input_size)` or `(batch, seq_len, input_size)` when
  `batch_first=True`.
- **Packed input**: `PackedSequence` with `data` shaped
  `(sum(batch_sizes), input_size)` and a 1D `batch_sizes` tensor. When a packed
  input is provided, `batch_first` has no effect.
- **Initial hidden state `hx`**:
    - For `RNN_TANH` / `RNN_RELU`: `(num_layers, batch, hidden_size)`. If omitted, a
      zero tensor is created. A 2D `(batch, hidden_size)` tensor is allowed and
      expanded to the first layer.
    - For `LSTM`: tuple `(h_0, c_0)`, each shaped `(num_layers, batch, hidden_size)`.
      A 2D tensor for either element is expanded similarly.
- **Output**: same leading dimensions as the input, with feature size `hidden_size`.
- **Packed output**: `PackedSequence` with the same `batch_sizes` and index metadata
  as the input.
- **Final hidden state**:
    - For simple RNN modes and GRU: `h_n` shaped `(num_layers, batch, hidden_size)`.
    - For `LSTM`: tuple `(h_n, c_n)`, each shaped `(num_layers, batch, hidden_size)`.

Examples
--------
**Running a single-layer tanh RNN over a sequence:**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> seq = lucid.randn(5, 2, 3)  # (seq_len=5, batch=2, input_size=3)
    >>> rnn = nn.RNNBase(mode="RNN_TANH", input_size=3, hidden_size=4)
    >>> output, h_n = rnn(seq)
    >>> output.shape
    (5, 2, 4)
    >>> h_n.shape
    (1, 2, 4)

**Using multiple layers, ReLU nonlinearity, dropout, and batch-first input:**

.. code-block:: python

    >>> seq = lucid.randn(2, 7, 8)  # (batch=2, seq_len=7, input_size=8)
    >>> rnn = nn.RNNBase(
    ...     mode="RNN_RELU",
    ...     input_size=8,
    ...     hidden_size=5,
    ...     num_layers=3,
    ...     dropout=0.1,
    ...     batch_first=True,
    ... )
    >>> output, h_n = rnn(seq)
    >>> output.shape  # matches batch-first layout
    (2, 7, 5)
    >>> h_n.shape  # one hidden state per layer
    (3, 2, 5)

**Providing an initial hidden state:**

.. code-block:: python

    >>> h0 = lucid.zeros(2, 2, 4)  # (num_layers=2, batch=2, hidden_size=4)
    >>> rnn = nn.RNNBase("RNN_TANH", input_size=4, hidden_size=4, num_layers=2)
    >>> seq = lucid.randn(6, 2, 4)
    >>> output, h_n = rnn(seq, h0)
    >>> (output.shape, h_n.shape)
    ((6, 2, 4), (2, 2, 4))

**LSTM mode with learnable biases and provided `(h_0, c_0)`:**

.. code-block:: python

    >>> h0 = lucid.zeros(2, 3, 6)
    >>> c0 = lucid.zeros(2, 3, 6)
    >>> rnn = nn.RNNBase("LSTM", input_size=8, hidden_size=6, num_layers=2, bias=True)
    >>> seq = lucid.randn(5, 3, 8)
    >>> output, (h_n, c_n) = rnn(seq, (h0, c0))
    >>> output.shape, h_n.shape, c_n.shape
    ((5, 3, 6), (2, 3, 6), (2, 3, 6))

**GRU mode with a single layer:**

.. code-block:: python

    >>> seq = lucid.randn(4, 1, 5)
    >>> gru = nn.RNNBase("GRU", input_size=5, hidden_size=3)
    >>> output, h_n = gru(seq)
    >>> output.shape, h_n.shape
    ((4, 1, 3), (1, 1, 3))
