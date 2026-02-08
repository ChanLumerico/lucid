lucid.random.randn
====================

.. autofunction:: lucid.random.randn

The `randn` function generates a tensor of the specified shape, filled with random 
values sampled from a standard normal distribution (mean = 0, variance = 1).

Function Signature
------------------

.. code-block:: python

    def randn(
        *shape: int,
        requires_grad: bool = False,
        keep_grad: bool = False,
        device: _DeviceType = "cpu",
    ) -> Tensor

Parameters
----------

- **shape** (*int*): The shape of the output tensor, specified as one or more integers.

- **requires_grad** (*bool*, optional): If `True`, the resulting tensor will track gradients 
  for automatic differentiation. Defaults to `False`.

- **keep_grad** (*bool*, optional): Determines whether gradient history should persist across 
  multiple operations. Defaults to `False`.

Returns
-------

- **Tensor**: A tensor of shape `shape` filled with random values drawn from a 
  standard normal distribution.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.random.randn(2, 3)
    >>> print(x)
    Tensor([[ 0.78357324, -0.26325684,  0.22752314],
            [-0.68811989, -1.01051757,  0.41808962]], grad=None)

By default, the generated tensor does not track gradients. 
To enable gradient tracking, set `requires_grad=True`:

.. code-block:: python

    >>> y = lucid.random.randn(3, 2, requires_grad=True)
    >>> print(y.requires_grad)
    True

.. note::

  - The random values are drawn from a standard normal distribution.
  - Use `lucid.random.seed` to ensure reproducibility of random values.
  - Since the output values are continuous, backpropagation through this function is not defined.
