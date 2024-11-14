lucid.tensor
============

.. autofunction:: lucid.tensor

Creates a new `Tensor` object with specified data, `requires_grad` setting, and data type. This function is similar to PyTorch's `torch.tensor`, allowing you to initialize a `Tensor` either from another `Tensor` or directly from array-like data.

Function Signature
------------------

.. code-block:: python

    def tensor(
        data: Tensor | _ArrayOrScalar, 
        requires_grad: bool = False, 
        dtype: Any = np.float32
    ) -> Tensor

Parameters
----------

- **data** (*Tensor | _ArrayOrScalar*):
  The input data to be wrapped in a `Tensor`. If a `Tensor` is provided, its underlying data is used.
  
- **requires_grad** (*bool*, optional):
  If True, the resulting `Tensor` will track gradients for backpropagation. Defaults to False.
  
- **dtype** (*Any*, optional):
  The desired data type for the new `Tensor`. Defaults to `np.float32`.

Returns
-------

- **Tensor**:
  A new `Tensor` instance wrapping the input data, with the specified `requires_grad` setting and data type.

Examples
--------

Create a `Tensor` from a list:

.. code-block:: python

    >>> t = tensor([1, 2, 3], requires_grad=True)
    >>> print(t)
    Tensor([1, 2, 3], requires_grad=True)

Use another `Tensor` as input:

.. code-block:: python

    >>> t1 = tensor([4.0, 5.0, 6.0])
    >>> t2 = tensor(t1, dtype=np.float64)
    >>> print(t2)
    Tensor([4.0, 5.0, 6.0], dtype=float64)
