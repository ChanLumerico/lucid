lucid.hstack
============

.. autofunction:: lucid.hstack

The `hstack` function stacks a sequence of tensors horizontally (column-wise) 
along the second dimension (axis=1 by default). 

It concatenates the input tensors along the specified axis, 
effectively increasing the size of that dimension.

Function Signature
------------------

.. code-block:: python

    def hstack(arr: tuple[Tensor, ...]) -> Tensor

Parameters
----------

- **arr** (*tuple[Tensor, ...]*): 
    A sequence of tensors to be horizontally stacked. All tensors must have the 
    same shape except in the dimension corresponding to `axis`.

Returns
-------

- **Tensor**: 
    A new tensor resulting from horizontally stacking the input tensors. 
    The resulting tensor has the same number of dimensions as the input tensors, 
    with the size along the horizontal axis (`axis=1` by default) being the sum 
    of the sizes of the input tensors along that axis.

Forward Calculation
-------------------

The `hstack` operation concatenates the input tensors along the horizontal axis.

.. math::

    \mathbf{out} = \text{hstack}(\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n)

Backward Gradient Calculation
-----------------------------

The gradient for the `hstack` operation is split and passed to each of the input 
tensors along the concatenated axis.

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}_i} = \mathbf{I}_i

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0, 2.0]], requires_grad=True)
    >>> b = Tensor([[3.0, 4.0]], requires_grad=True)
    >>> h_stacked = lucid.hstack((a, b))
    >>> print(h_stacked)
    Tensor([[1. 2. 3. 4.]], grad=None)

    >>> c = Tensor([[5.0, 6.0]], requires_grad=True)
    >>> h_stacked = lucid.hstack((a, b, c))
    >>> print(h_stacked)
    Tensor([[1. 2. 3. 4. 5. 6.]], grad=None)

.. note::

    - All input tensors must have the same shape except in the dimension 
      corresponding to the horizontal axis (`axis=1` by default).
      
    - The `hstack` operation increases the size of the specified axis by concatenating the input tensors.
    - The resulting tensor will have the same number of dimensions as the input tensors.
    - If any of the input tensors require gradients, the resulting tensor will also require gradients.
    - The horizontal axis can be adjusted by modifying the underlying implementation if necessary.