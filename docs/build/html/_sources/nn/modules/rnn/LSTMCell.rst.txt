nn.LSTMCell
===========

.. autoclass:: lucid.nn.LSTMCell

`LSTMCell` performs a single time-step update of a long short-term memory unit.
It uses input, forget, candidate, and output gates to update the hidden and cell
states. Both unbatched (`(input_size,)`) and batched (`(batch_size, input_size)`)
inputs are supported.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.LSTMCell(
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    )

Parameters
----------
- **input_size** (*int*):
  Number of expected features in the input `x_t`.

- **hidden_size** (*int*):
  Number of features in both the hidden state `h_t` and cell state `c_t`.

- **bias** (*bool*, optional):
  If `True`, adds learnable biases for input-to-hidden and hidden-to-hidden
  projections. Default: `True`.

Attributes
----------
- **weight_ih** (*Tensor*):
  Input-to-hidden weight of shape `(4 * hidden_size, input_size)`.

- **weight_hh** (*Tensor*):
  Hidden-to-hidden weight of shape `(4 * hidden_size, hidden_size)`.

- **bias_ih** (*Tensor* or *None*):
  Input bias; `None` when `bias=False`.

- **bias_hh** (*Tensor* or *None*):
  Hidden-state bias; `None` when `bias=False`.

Forward Calculation
-------------------
Given input :math:`x_t` and previous states :math:`(h_{t-1}, c_{t-1})`, the cell
computes:

.. math::

    \begin{aligned}
    [i_t, f_t, g_t, o_t] &= x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh} \\
    i_t &= \sigma(i_t), \quad f_t = \sigma(f_t), \quad g_t = \tanh(g_t), \quad o_t = \sigma(o_t) \\
    c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
    h_t &= o_t \odot \tanh(c_t)
    \end{aligned}

Handling Initial State
----------------------
- If `hx` is not provided, both `h_t` and `c_t` are initialized to zeros on the
  same device/dtype as the input.
- Inputs and states may be 1D (unbatched) or 2D (batched). Shapes must match on
  `batch_size` and `hidden_size` or a `ValueError` is raised.
- When given unbatched input, the returned states are also unbatched (batch
  dimension squeezed).

Examples
--------
**Single step with default initialization:**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x_t = lucid.randn(5)  # (input_size=5)
    >>> cell = nn.LSTMCell(input_size=5, hidden_size=3)
    >>> h_t, c_t = cell(x_t)
    >>> h_t.shape, c_t.shape
    ((3,), (3,))

**Batched step with provided hidden and cell states:**

.. code-block:: python

    >>> x_t = lucid.randn(4, 6)       # (batch=4, input_size=6)
    >>> h0 = lucid.zeros(4, 8)        # (batch, hidden_size)
    >>> c0 = lucid.zeros(4, 8)
    >>> cell = nn.LSTMCell(input_size=6, hidden_size=8, bias=False)
    >>> h_t, c_t = cell(x_t, (h0, c0))
    >>> h_t.shape, c_t.shape
    ((4, 8), (4, 8))
