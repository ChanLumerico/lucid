lucid.reshape
=============

.. autofunction:: lucid.reshape

The `reshape` function changes the shape of a tensor while keeping the same data. 
It does not alter the underlying data, only the way it is viewed.

Function Signature
------------------

.. code-block:: python

    def reshape(a: Tensor, shape: _ShapeLike) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor to reshape.

- **shape** (*_ShapeLike*): The new shape for the tensor. 
  This can be a list or tuple of integers.

Returns
-------

- **Tensor**: 
    A new tensor with the specified shape. 
    If the input tensor **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward operation simply reshapes the tensor without changing its underlying data. 
The total number of elements in the new shape must match the original tensor.

.. math::

    \mathbf{out} = \text{reshape}(\mathbf{a}, \text{shape})

Backward Gradient Calculation
-----------------------------

The gradient for reshaping is the identity function for the restructured tensor. 
There is no change in the values of the tensor elements, only the way they are indexed. 
Hence, the backward pass follows the same shape.

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \mathbf{I}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
    >>> reshaped = lucid.reshape(a, (2, 3))  # or a.reshape(2, 3)
    >>> print(reshaped)
    Tensor([[1. 2. 3.] [4. 5. 6.]], grad=None)

.. note::

    - The reshape operation does not change the total number of elements in the tensor, 
      but rather the shape in which they are accessed.

    - The operation supports broadcasting and views.

    - Gradients are propagated through the reshaped tensor in the same way as the original tensor.
