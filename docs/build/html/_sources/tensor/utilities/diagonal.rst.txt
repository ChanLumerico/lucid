lucid.diagonal
==============

.. autofunction:: lucid.diagonal

The `diagonal` function extracts the diagonal elements from the input tensor
along two specified axes, optionally offset from the main diagonal.

Function Signature
------------------

.. code-block:: python

    def diagonal(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor

Parameters
----------

- **a** (*Tensor*):
  The input tensor from which to extract the diagonal.

- **offset** (*int*, optional):
  Offset from the main diagonal. Defaults to `0`.
    
  - A positive value selects diagonals above the main diagonal.
  - A negative value selects diagonals below the main diagonal.

- **axis1** (*int*, optional):
  The first axis to consider as part of the 2D plane from which 
  to take the diagonal. Defaults to `0`.

- **axis2** (*int*, optional):
  The second axis to consider as part of the 2D plane from which 
  to take the diagonal. Defaults to `1`.

Returns
-------

- **Tensor**:
  A one-dimensional tensor containing the selected diagonal values.

Mathematics
-----------

.. math::

    \text{diag}(A) = \left\{ A[i_0, \dots, i_{\text{axis1}} = k, 
    \dots, i_{\text{axis2}} = k + \text{offset}, \dots] \right\}_{k}

Examples
--------

Extracting a diagonal from a 2D tensor:

.. code-block:: python

    >>> import lucid
    >>> a = lucid.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True)
    >>> d = lucid.diagonal(a)
    >>> print(d)
    Tensor([1, 5, 9])

    >>> d.sum().backward()
    >>> print(a.grad)
    [[1. 0. 0.],
     [0. 1. 0.],
     [0. 0. 1.]]

.. note::

    The gradient is propagated only to the diagonal entries.
