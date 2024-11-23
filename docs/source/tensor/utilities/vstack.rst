lucid.vstack
============

.. autofunction:: lucid.vstack

The `vstack` function stacks a sequence of tensors vertically (row-wise) 
along the first dimension (axis=0 by default). 

It concatenates the input tensors along the specified axis, 
effectively increasing the size of that dimension.

Function Signature
------------------

.. code-block:: python

    def vstack(arr: tuple[Tensor, ...]) -> Tensor

Parameters
----------

- **arr** (*tuple[Tensor, ...]*): 
    A sequence of tensors to be vertically stacked. All tensors must have the 
    same shape except in the dimension corresponding to `axis`.

Returns
-------

- **Tensor**: 
    A new tensor resulting from vertically stacking the input tensors. 
    The resulting tensor has the same number of dimensions as the input tensors, 
    with the size along the vertical axis (`axis=0` by default) being the sum of 
    the sizes of the input tensors along that axis.

Forward Calculation
-------------------

The `vstack` operation concatenates the input tensors along the vertical axis.

.. math::

    \mathbf{out} = \text{vstack}(\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n)

Backward Gradient Calculation
-----------------------------

The gradient for the `vstack` operation is split and passed to each of the 
input tensors along the concatenated axis.

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}_i} = \mathbf{I}_i

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0, 2.0]], requires_grad=True)
    >>> b = Tensor([[3.0, 4.0]], requires_grad=True)
    >>> v_stacked = lucid.vstack((a, b))
    >>> print(v_stacked)
    Tensor([[1. 2.],
            [3. 4.]], grad=None)

    >>> c = Tensor([[5.0, 6.0]], requires_grad=True)
    >>> v_stacked = lucid.vstack((a, b, c))
    >>> print(v_stacked)
    Tensor([[1. 2.],
            [3. 4.],
            [5. 6.]], grad=None)

.. note::

    - All input tensors must have the same shape except in the dimension corresponding 
      to the vertical axis (`axis=0` by default).

    - The `vstack` operation increases the size of the specified axis by concatenating the input tensors.
    - The resulting tensor will have the same number of dimensions as the input tensors.
    - If any of the input tensors require gradients, the resulting tensor will also require gradients.
    - The vertical axis can be adjusted by modifying the underlying implementation if necessary.