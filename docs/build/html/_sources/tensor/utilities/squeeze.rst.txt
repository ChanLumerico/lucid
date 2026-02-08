lucid.squeeze
=============

.. autofunction:: lucid.squeeze

The `squeeze` function removes single-dimensional entries from the shape of a tensor. 
It is commonly used to remove dimensions of size 1 from the tensor’s shape.

Function Signature
------------------

.. code-block:: python

    def squeeze(a: Tensor, axis: _ShapeLike | None = None) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor to squeeze.
- **axis** (*_ShapeLike* | *None*, optional): 
    The dimensions to remove. 
    If `None`, all dimensions of size 1 will be removed. 
    If specified, only the listed axes with size 1 will be removed.

Returns
-------

- **Tensor**: 
    A tensor with the specified dimensions removed. 
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The `squeeze` operation removes dimensions with a size of 1 from the tensor.

.. math::

    \mathbf{out}_i = \text{squeeze}(\mathbf{a}_i)

Backward Gradient Calculation
-----------------------------

For a squeezed tensor, the gradient is passed through unchanged, 
as the operation only affects the shape and not the data.

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \mathbf{I}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    >>> squeezed = lucid.squeeze(a)  # or a.squeeze()
    >>> print(squeezed)
    Tensor([1. 2. 3.], grad=None)

.. note::

    - If `axis` is provided, it will only remove the specified dimensions of size 1.
    - The resulting tensor will have the same number of elements but a potentially different shape.
