lucid.random.rand
=================

.. autofunction:: lucid.random.rand

The `rand` function generates a tensor of the specified shape, 
filled with random values drawn from a uniform distribution over the 
interval :math:`[0, 1)`.

Function Signature
------------------

.. code-block:: python

    def rand(
      *shape: int, 
      requires_grad: bool = False, 
      keep_grad: bool = False,
      device: _DeviceType = "cpu",
    ) -> Tensor

Parameters
----------

- **shape** (*int*): The dimensions of the tensor to generate. 
  Accepts a variable number of arguments for multidimensional tensors.

- **requires_grad** (*bool*, optional): If set to `True`, the resulting tensor 
  will track gradients for automatic differentiation. Defaults to `False`.

- **keep_grad** (*bool*, optional): Determines whether gradient history should 
  persist across multiple operations. Defaults to `False`.

Returns
-------

- **Tensor**: A tensor of shape `shape` with random values uniformly 
  distributed in :math:`[0, 1)`.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.random.rand(2, 3)
    >>> print(x)
    Tensor([[0.64589411, 0.43758721, 0.891773  ],
            [0.96366276, 0.38344152, 0.79172504]], grad=None)

By default, the generated tensor does not track gradients. 
Set `requires_grad=True` to enable gradient tracking:

.. code-block:: python

    >>> y = lucid.random.rand(3, 2, requires_grad=True)
    >>> print(y.requires_grad)
    True

.. note::

    - The random values are drawn from a uniform distribution, which is suitable 
      for initializing weights or general-purpose random number generation.

    - Use `lucid.random.seed` to ensure reproducibility of random values.