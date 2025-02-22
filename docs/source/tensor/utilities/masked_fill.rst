lucid.masked_fill
=================

.. autofunction:: lucid.masked_fill

The `masked_fill` function replaces elements in a tensor where a given mask 
is `True` with a specified scalar value. This function is useful for setting 
specific entries in a tensor to a constant while ensuring correct gradient 
computation.

Function Signature
------------------

.. code-block:: python

    def masked_fill(self: Tensor, mask: Tensor, value: _Scalar) -> Tensor

Parameters
----------

- **self** (*Tensor*):
  The input tensor.

- **mask** (*Tensor*):
  A boolean tensor of the same shape as `self`. Elements marked as `True` 
  will be replaced with `value`.

- **value** (*_Scalar*):
  The scalar value to assign to elements where `mask` is `True`.

Returns
-------

- **Tensor**: 
  A new tensor where elements in `self` have been replaced with `value` 
  wherever `mask` is `True`. The output tensor retains gradient tracking if 
  `self` requires gradients.

Gradient Computation
--------------------

During backpropagation, the gradient is computed as:

.. math::

    \frac{\partial \text{output}}{\partial \text{self}} = \begin{cases}
        0, & \text{if mask is True} \\
        1, & \text{otherwise}
    \end{cases}

This ensures that no gradients flow to positions replaced by `value`.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> mask = lucid.Tensor([[True, False], [False, True]])
    >>> y = lucid.masked_fill(x, mask, -1.0)
    >>> print(y)
    Tensor([[-1.0,  2.0],
            [ 3.0, -1.0]])
    
    # Backpropagation
    >>> y.backward()
    >>> print(x.grad)
    [[0.0, 1.0],
     [1.0, 0.0]]

.. note::

    - The `mask` tensor must have the same shape as `self`.
    - This function does **not** modify `self` in-place; it returns a new tensor.
    - The gradient of positions set by the mask is always `0` during backpropagation.
