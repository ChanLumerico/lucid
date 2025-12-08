nn.RNNCell
==========

.. autoclass:: lucid.nn.RNNCell

The `RNNCell` module implements a single recurrent step. It processes one time step
of input and combines it with the previous hidden state using a configurable
activation function (`tanh` or `relu`). The cell supports both unbatched inputs
(`(input_size,)`) and batched inputs (`(batch_size, input_size)`).

Class Signature
---------------
.. code-block:: python

    class lucid.nn.RNNCell(
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
    )

Parameters
----------
- **input_size** (*int*):
  Number of expected features in the input `x_t`.

- **hidden_size** (*int*):
  Number of features in the hidden state `h_t`.

- **bias** (*bool*, optional):
  If `True`, adds learnable bias terms `bias_ih` and `bias_hh`.
  Default: `True`.

- **nonlinearity** (*Literal["tanh", "relu"]*, optional):
  Activation applied to the combined input and hidden projection.
  Default: `"tanh"`.

Attributes
----------
- **weight_ih** (*Tensor*):
  Input-to-hidden weight of shape `(hidden_size, input_size)`.

- **weight_hh** (*Tensor*):
  Hidden-to-hidden weight of shape `(hidden_size, hidden_size)`.

- **bias_ih** (*Tensor* or *None*):
  Bias added to the input projection. `None` when `bias=False`.

- **bias_hh** (*Tensor* or *None*):
  Bias added to the hidden-state projection. `None` when `bias=False`.

- **nonlinearity** (*Module*):
  Activation module instance (`nn.Tanh` or `nn.ReLU`) applied elementwise.

Forward Calculation
-------------------
For an input :math:`x_t` and previous hidden state :math:`h_{t-1}`, the cell
computes:

.. math::

    h_t = \sigma(x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})

Where:

- :math:`\sigma` is either :math:`\tanh` or :math:`\text{ReLU}`.
- :math:`x_t` has shape `(input_size)` or `(batch_size, input_size)`.
- :math:`h_{t-1}` has shape `(hidden_size)` or `(batch_size, hidden_size)`.

Handling Initial State
----------------------
- If `hx` is not provided, the hidden state is initialized to zeros on the same
  device and dtype as the input.

- Inputs and hidden states can be 1D (unbatched) or 2D (batched). Shapes must
  agree on `batch_size` and `hidden_size`; otherwise, a `ValueError` is raised.

- When receiving unbatched input, the returned hidden state is also unbatched
  (the batch dimension is squeezed out).

Examples
--------
**Single step without an initial hidden state:**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x_t = lucid.Tensor([0.5, -1.0, 0.3], requires_grad=True)  # Shape: (3,)
    >>> cell = nn.RNNCell(input_size=3, hidden_size=2, nonlinearity="tanh")
    >>> h_t = cell(x_t)  # hx defaults to zeros
    >>> h_t.shape
    (2,)

**Iterating over a sequence manually:**

.. code-block:: python

    >>> seq = lucid.Tensor([
    ...     [0.1, 0.2],
    ...     [0.0, -0.4],
    ...     [0.3, 0.5],
    ... ], requires_grad=True)
    >>> cell = nn.RNNCell(input_size=2, hidden_size=3, nonlinearity="relu")
    >>> h = None
    >>> for x_t in seq:
    ...     h = cell(x_t, h)
    >>> h  # Final hidden state after the last time step
    Tensor([...], grad=None)

**Batched inputs with an explicit initial state:**

.. code-block:: python

    >>> batch = lucid.Tensor([[1.0, -0.2], [0.4, 0.6]], requires_grad=True)  # Shape: (2, 2)
    >>> h0 = lucid.zeros(2, 4)  # Shape: (batch_size, hidden_size)
    >>> cell = nn.RNNCell(input_size=2, hidden_size=4, bias=False)
    >>> h1 = cell(batch, h0)
    >>> h1.shape
    (2, 4)
