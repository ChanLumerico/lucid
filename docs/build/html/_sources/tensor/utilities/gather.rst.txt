lucid.gather
============

.. autofunction:: lucid.gather

The `gather` function selects values from `input_` along a single axis (`dim`)
using integer indices from `index`, following PyTorch-style gather semantics.

Function Signature
------------------

.. code-block:: python

    def gather(input_: Tensor, dim: int, index: Tensor) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  Source tensor to gather values from.

- **dim** (*int*):
  Axis along which to gather. Negative values are allowed.

- **index** (*Tensor*):
  Integer tensor of indices. Output shape is exactly `index.shape`.

Shape Rules
-----------

- `index.ndim` must equal `input_.ndim`.
- For every axis `d != dim`, `index.size(d) <= input_.size(d)`.
- Every index value must satisfy `0 <= index[...,] < input_.size(dim)`.
- Output shape is `index.shape`.

2D semantics:

.. math::

    \text{if } dim = 0,\quad out[i, j] = input[index[i, j], j]

    \text{if } dim = 1,\quad out[i, j] = input[i, index[i, j]]

Gradient Computation
--------------------

Gradients are accumulated back into `input_` at gathered locations
(scatter-add behavior). Repeated indices accumulate gradients.
`index` is non-differentiable and receives zero gradient.

.. math::

    \frac{\partial \text{gather}(x, dim, idx)}{\partial x}
    = \text{scatter\_add}(\text{grad\_out}, idx, dim),
    \qquad
    \frac{\partial \text{gather}}{\partial idx} = 0

Example
-------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.tensor([[10, 20, 30],
    ...                   [40, 50, 60]], requires_grad=True)
    >>> idx = lucid.tensor([[2, 1],
    ...                     [0, 2]], dtype=lucid.Int32)
    >>> out = lucid.gather(x, dim=1, index=idx)
    >>> print(out)
    [[30 20]
     [40 60]]
    >>> out.sum().backward()
    >>> print(x.grad)
    [[0 1 1]
     [1 0 1]]

.. note::

   `gather` does not interpret `index` entries as full coordinates.
   Each index value selects only along `dim`; other coordinates come
   from the output position.
