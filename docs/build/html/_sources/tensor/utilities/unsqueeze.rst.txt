lucid.unsqueeze
===============

.. autofunction:: lucid.unsqueeze

The `unsqueeze` function adds a dimension of size 1 at the specified axis, 
which can be useful for broadcasting in operations.

Function Signature
------------------

.. code-block:: python

    def unsqueeze(a: Tensor, axis: _ShapeLike) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor to unsqueeze.
- **axis** (*_ShapeLike*): The axis along which to add the new dimension of size 1.

Returns
-------

- **Tensor**: 
    A tensor with an added dimension of size 1. 
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The `unsqueeze` operation adds a new dimension of size 1 at the specified axis.

.. math::

    \mathbf{out}_i = \text{unsqueeze}(\mathbf{a}_i, \text{axis})

Backward Gradient Calculation
-----------------------------

For an unsqueezed tensor, the gradient is propagated through the new dimension 
without any changes to the values of the original tensor.

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \mathbf{I}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> unsqueezed = lucid.unsqueeze(a, 0)  # or a.unsqueeze(0)
    >>> print(unsqueezed)
    Tensor([[1. 2. 3.]], grad=None)

.. note::

    - The `unsqueeze` operation is typically used to add a dimension to match 
      broadcasting requirements in operations.
