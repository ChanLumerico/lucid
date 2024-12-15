lucid.empty_like
=================

.. autofunction:: lucid.empty_like

The `empty_like` function generates a tensor with the same shape as the input tensor,
with uninitialized values in its memory. The data type and gradient tracking behavior
can be specified independently of the input tensor.

Function Signature
------------------

.. code-block:: python

    def empty_like(
        a: Tensor | _ArrayLike,
        dtype: type | None = None,
        requires_grad: bool = False,
        keep_grad: bool = False
    ) -> Tensor

Parameters
----------

- **a** (*Tensor* or *_ArrayLike*): The input tensor whose shape is used to define
  the shape of the output tensor.

- **dtype** (*type*, optional): The data type of the uninitialized tensor. If `None`,
  the data type of `a` is used. Defaults to `None`.

- **requires_grad** (*bool*, optional): If set to `True`, the resulting tensor
  will track gradients for automatic differentiation. Defaults to `False`.

- **keep_grad** (*bool*, optional): Determines whether gradient history should
  persist across multiple operations. Defaults to `False`.

Returns
-------

- **Tensor**: A tensor with the same shape as `a`, containing uninitialized values.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1, 2, 3], [4, 5, 6]])
    >>> x = lucid.empty_like(a)
    >>> print(x)
    Tensor([[ 6.949e-310,  6.949e-310,  0.000e+000],
            [ 0.000e+000,  0.000e+000,  0.000e+000]], grad=None)

The memory is uninitialized, so values are arbitrary. 
To track gradients, set `requires_grad=True`:

.. code-block:: python

    >>> y = lucid.empty_like(a, requires_grad=True)
    >>> print(y.requires_grad)
    True

You can also specify the data type of the new tensor:

.. code-block:: python

    >>> z = lucid.empty_like(a, dtype=float)
    >>> print(z.dtype)
    float

.. note::

    - The `empty_like` function does not clear the memory where the tensor is allocated,
      so the values in the tensor are arbitrary.
      If you need a tensor with zeros, consider using `lucid.zeros_like` instead.
