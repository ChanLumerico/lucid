lucid.random.randint
====================

.. autofunction:: lucid.random.randint

The `randint` function generates a tensor of the specified shape, 
filled with random integer values from a specified range.

Function Signature
------------------

.. code-block:: python

    def randint(
        low: int,
        high: int | None,
        size: int | _ShapeLike,
        requires_grad: bool = False,
        keep_grad: bool = False,
    ) -> Tensor

Parameters
----------

- **low** (*int*): The lowest integer (inclusive) in the range from which random 
  values will be drawn.

- **high** (*int*, optional): The highest integer (exclusive) in the range. 
  If `None`, the random values will be drawn from the range :math:`[0, \text{low})`.

- **size** (*int* or *ShapeLike*): The shape of the output tensor.

- **requires_grad** (*bool*, optional): If `True`, the resulting tensor will track 
  gradients for automatic differentiation. Defaults to `False`.

- **keep_grad** (*bool*, optional): Determines whether gradient history should persist 
  across multiple operations. Defaults to `False`.

Returns
-------

- **Tensor**: A tensor of shape `size` filled with random integers drawn from the specified range.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.random.randint(0, 10, (2, 3))
    >>> print(x)
    Tensor([[9, 7, 3],
            [7, 8, 2]], grad=None)

By default, the generated tensor does not track gradients. Set `requires_grad=True` to enable gradient tracking:

.. code-block:: python

    >>> y = lucid.random.randint(1, 5, (3, 2), requires_grad=True)
    >>> print(y.requires_grad)
    True

.. note::

  - The random values are drawn from a discrete uniform distribution.
  - Use `lucid.random.seed` to ensure reproducibility of random values.
  - The gradient calculation for this operation is not defined because the output is discrete.
