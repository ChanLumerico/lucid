lucid.roll
==========

.. autofunction:: lucid.roll

The `roll` function shifts elements of a tensor along specified dimensions. 
This operation does not alter the tensor's shape, only reorders elements cyclically.

Function Signature
------------------

.. code-block:: python

    def roll(self: Tensor, shifts: int | tuple[int, ...], dims: int | tuple[int, ...] = None) -> Tensor

Parameters
----------

- **self** (*Tensor*):
  The input tensor.

- **shifts** (*int | tuple[int, ...]*):
  The number of places by which elements are shifted. If a tuple, 
  each entry corresponds to a dimension in `dims`.

- **dims** (*int | tuple[int, ...] | None, optional*):
  The dimensions along which elements are shifted. If `None`, 
  the tensor is flattened before rolling. Default is `None`.

Returns
-------

- **Tensor**:
  A new tensor with elements shifted along the specified dimensions.

Gradient Computation
--------------------

During backpropagation, the gradient is computed as:

.. math::

    \frac{\partial \text{output}}{\partial \text{self}} = 
    \text{roll}(\text{grad}, -\text{shifts}, \text{dims})

This ensures that gradients are shifted in the opposite direction.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    >>> y = lucid.roll(x, shifts=1, dims=1)
    >>> print(y)
    Tensor([[3, 1, 2],
            [6, 4, 5]])
    
    # Backpropagation
    >>> y.backward()
    >>> print(x.grad)
    [[2, 3, 1],
     [5, 6, 4]]

.. note::

    - If `dims` is `None`, `roll` acts on a flattened version of the tensor.
    - The function does **not** modify `self` in-place; it returns a new tensor.
    - This operation preserves the original tensor's shape.
