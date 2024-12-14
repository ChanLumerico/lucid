lucid.random.uniform
====================

.. autofunction:: lucid.random.uniform

The `uniform` function generates a tensor of the specified shape,
filled with random values drawn from a uniform distribution over the
interval :math:`[low, high)`.

Function Signature
------------------

.. code-block:: python

    def uniform(
        low: float,
        high: float,
        size: int | tuple[int, ...],
        requires_grad: bool = False,
        keep_grad: bool = False
    ) -> Tensor

Parameters
----------

- **low** (*float*): The lower bound of the interval from which the random
  values are drawn (inclusive).

- **high** (*float*): The upper bound of the interval from which the random
  values are drawn (exclusive).

- **size** (*int* or *tuple of int*): The dimensions of the tensor to generate.
  Can be a single integer for a 1D tensor or a tuple for multidimensional tensors.

- **requires_grad** (*bool*, optional): If set to `True`, the resulting tensor
  will track gradients for automatic differentiation. Defaults to `False`.

- **keep_grad** (*bool*, optional): Determines whether gradient history should
  persist across multiple operations. Defaults to `False`.

Returns
-------

- **Tensor**: A tensor of shape `size` with random values uniformly
  distributed in :math:`[low, high)`.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.random.uniform(0, 10, (2, 3))
    >>> print(x)
    Tensor([[2.345, 7.654, 5.123],
            [6.234, 1.543, 9.876]], grad=None)

By default, the generated tensor does not track gradients.
Set `requires_grad=True` to enable gradient tracking:

.. code-block:: python

    >>> y = lucid.random.uniform(-1, 1, (3, 2), requires_grad=True)
    >>> print(y.requires_grad)
    True

.. note::

    - The random values are drawn from a uniform distribution over the interval :math:`[low, high)`,
      which is useful for parameter initialization in neural networks and general-purpose random number generation.

    - Use `lucid.random.seed` to ensure reproducibility of random values.

