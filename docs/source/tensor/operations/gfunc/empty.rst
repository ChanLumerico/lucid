lucid.empty
===========

.. autofunction:: lucid.empty

The `empty` function generates a tensor of the specified shape, 
with uninitialized values in its memory. This function does not set 
the values to zero or any other specific value.

Function Signature
------------------

.. code-block:: python

    def empty(
        *args: int | tuple[int, ...], 
        dtype: type = _base_dtype, 
        requires_grad: bool = False, 
        keep_grad: bool = False,
        device: _DeviceType = "cpu",
    ) -> Tensor

Parameters
----------

- **shape** (*int* or *tuple of int*): The dimensions of the tensor to generate. 
  Can be a variable number of integer arguments for multidimensional tensors 
  or a single tuple specifying the shape.

- **dtype** (*type*, optional): The data type of the uninitialized tensor. Defaults to `_base_dtype`.

- **requires_grad** (*bool*, optional): If set to `True`, the resulting tensor 
  will track gradients for automatic differentiation. Defaults to `False`.

- **keep_grad** (*bool*, optional): Determines whether gradient history should 
  persist across multiple operations. Defaults to `False`.

Returns
-------

- **Tensor**: A tensor of the specified `shape` with uninitialized values.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.empty(2, 3)
    >>> print(x)
    Tensor([[ 6.949e-310,  6.949e-310,  0.000e+000],
            [ 0.000e+000,  0.000e+000,  0.000e+000]], grad=None)

The memory is uninitialized, so values are arbitrary. 
To track gradients, set `requires_grad=True`:

.. code-block:: python

    >>> y = lucid.empty((3, 2), requires_grad=True)
    >>> print(y.requires_grad)
    True

.. note::

    - The `empty` function does not clear the memory where the tensor is allocated, 
      so the values in the tensor are arbitrary. 
      If you need a tensor with zeros, consider using `lucid.zeros` instead.
