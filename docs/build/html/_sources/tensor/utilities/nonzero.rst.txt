lucid.nonzero
=============

.. autofunction:: lucid.nonzero

The `nonzero` function returns the indices of non-zero elements in the input tensor. 
It outputs a 2D tensor where each row corresponds to the index of a non-zero element.

Function Signature
------------------

.. code-block:: python

    def nonzero(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor from which to find non-zero indices.

Returns
-------

- **Tensor**:
  A 2D tensor of shape :math:`(N, D)` where :math:`N` is the number of non-zero elements 
  and :math:`D` is the number of dimensions in the input tensor.

  Each row of the output represents a coordinate index of a non-zero element.

  The output tensor has `int32` dtype and does not track gradients.

.. note::

   This operation is non-differentiable. Gradients will not be propagated through `nonzero()`.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.tensor([[1, 0], [0, 2]])
    >>> lucid.nonzero(x)
    Tensor([[0, 0],
            [1, 1]], grad=None, device=cpu)

    >>> x = lucid.tensor([0, 3, 0, 4])
    >>> lucid.nonzero(x)
    Tensor([[1],
            [3]], grad=None, device=cpu)
