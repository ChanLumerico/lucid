nn.GRUCell
==========

.. autoclass:: lucid.nn.GRUCell

`GRUCell` performs a single time-step update of a gated recurrent unit. It uses
reset and update gates to mix the previous hidden state with the new candidate.
Both unbatched (`(input_size,)`) and batched (`(batch_size, input_size)`) inputs
are supported.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.GRUCell(
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    )

Parameters
----------
- **input_size** (*int*):
  Number of expected features in the input `x_t`.

- **hidden_size** (*int*):
  Number of features in the hidden state `h_t`.

- **bias** (*bool*, optional):
  If `True`, adds learnable biases for input-to-hidden and hidden-to-hidden
  projections. Default: `True`.

Attributes
----------
- **weight_ih** (*Tensor*):
  Input-to-hidden weight of shape `(3 * hidden_size, input_size)`.

- **weight_hh** (*Tensor*):
  Hidden-to-hidden weight of shape `(3 * hidden_size, hidden_size)`.

- **bias_ih** (*Tensor* or *None*):
  Input bias; `None` when `bias=False`.

- **bias_hh** (*Tensor* or *None*):
  Hidden-state bias; `None` when `bias=False`.

Forward Calculation
-------------------
Given input :math:`x_t` and previous hidden state :math:`h_{t-1}`, the cell
computes reset, update, and candidate activations:

.. math::

    \begin{aligned}
    [r_t, z_t, n_t] &= x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh} \\
    r_t &= \sigma(r_t), \quad z_t = \sigma(z_t), \quad n_t = \tanh(n_t + r_t \odot h_{t-1} W_{nh}^T) \\
    h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1}
    \end{aligned}

Handling Initial State
----------------------
- If `hx` is not provided, the hidden state is initialized to zeros on the same
  device/dtype as the input.
- Inputs and hidden states may be 1D (unbatched) or 2D (batched). Shapes must
  match on `batch_size` and `hidden_size` or a `ValueError` is raised.
- When given unbatched input, the returned hidden state is also unbatched (batch
  dimension squeezed).

Examples
--------
**Single step with default initialization:**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x_t = lucid.randn(3)  # (input_size=3)
    >>> cell = nn.GRUCell(input_size=3, hidden_size=4)
    >>> h_t = cell(x_t)
    >>> h_t.shape
    (4,)

**Batched step with provided hidden state:**

.. code-block:: python

    >>> x_t = lucid.randn(5, 2)  # (batch=5, input_size=2)
    >>> h0 = lucid.zeros(5, 6)
    >>> cell = nn.GRUCell(input_size=2, hidden_size=6, bias=False)
    >>> h_t = cell(x_t, h0)
    >>> h_t.shape
    (5, 6)
