nn.functional.normalize
=======================

.. autofunction:: lucid.nn.functional.normalize

The `normalize` function applies Lp-normalization to the input tensor 
along the specified dimension.

Function Signature
------------------

.. code-block:: python

    def normalize(input: Tensor, p: int = 2, dim: int = 1, eps: float = 1e-12) -> Tensor

Parameters
----------

- **input** (*Tensor*):
  The input tensor to be normalized.
- **p** (*int*, optional):
  The exponent value in the norm formulation. Default is 2 (Euclidean norm).
- **dim** (*int*, optional):
  The dimension along which to normalize. Default is 1.
- **eps** (*float*, optional):
  A small value to avoid division by zero. Default is 1e-12.

Returns
-------

- **Tensor**:
  The normalized tensor with the same shape as the input.

Mathematical Definition
-----------------------

For a given input tensor :math:`x`, the function computes the 
normalized tensor :math:`y` as:

.. math::

    y_i = \frac{x_i}{\max(\|x\|_p, \varepsilon)}

where:

- :math:`\|x\|_p = \left( \sum |x_i|^p \right)^{\frac{1}{p}}`
- :math:`\varepsilon` is a small constant to prevent division by zero.

Examples
--------

**L2 Normalization along a specified dimension:**

.. code-block:: python

    >>> import lucid
    >>> input_tensor = lucid.Tensor([[3.0, 4.0], [1.0, 2.0]])
    >>> output = lucid.nn.functional.normalize(input_tensor, p=2, dim=1)
    >>> print(output)
    Tensor([[0.6, 0.8],
            [0.4472, 0.8944]])

**L1 Normalization:**

.. code-block:: python

    >>> output = lucid.nn.functional.normalize(input_tensor, p=1, dim=1)
    >>> print(output)
    Tensor([[0.4286, 0.5714],
            [0.3333, 0.6667]])

.. note::

    The function ensures numerical stability by avoiding division by 
    zero using :math:`\max(\|x\|_p, \varepsilon)`.

.. caution::

    Ensure the input tensor does not have all-zero values along the specified dimension, 
    as it may lead to unexpected behavior despite the epsilon safeguard.
