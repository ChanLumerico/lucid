lucid.stack
===========

.. autofunction:: lucid.stack

The `stack` function concatenates a sequence of tensors along a new dimension. 
It stacks the input tensors along the specified axis, creating a new tensor 
with one more dimension than the input tensors.

Function Signature
------------------

.. code-block:: python

    def stack(arr: tuple[Tensor, ...], axis: int = 0) -> Tensor

Parameters
----------

- **arr** (*Tensor*): 
    A tuple of tensors to be stacked. All tensors must have the same shape.
    
- **axis** (*int*, optional): 
    The axis along which to stack the tensors. Defaults to `0`.

Returns
-------

- **Tensor**: 
    A new tensor resulting from stacking the input tensors along the specified axis. 
    The resulting tensor has one additional dimension compared to the input tensors.

Forward Calculation
-------------------

The `stack` operation concatenates the input tensors along a new dimension specified by `axis`.

.. math::

    \mathbf{out}[i, j, \dots] = \mathbf{arr}[i][j, \dots]

Backward Gradient Calculation
-----------------------------

The gradient for the stack operation is distributed to each of the input tensors along the stacked axis.

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{arr}_k} = \mathbf{I}_k

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1.0, 2.0], requires_grad=True)
    >>> b = Tensor([3.0, 4.0], requires_grad=True)
    >>> stacked = lucid.stack((a, b), axis=0)
    >>> print(stacked)
    Tensor([[1. 2.],
            [3. 4.]], grad=None)
    
    >>> stacked = lucid.stack((a, b), axis=1)
    >>> print(stacked)
    Tensor([[1. 3.],
            [2. 4.]], grad=None)

.. note::

    - All input tensors must have the same shape.
    - The `axis` parameter specifies the new dimension; for example, `axis=0` 
      stacks tensors vertically, while `axis=1` stacks them horizontally.
      
    - The resulting tensor will have one more dimension than the input tensors.
    - If any of the input tensors require gradients, the resulting tensor will also require gradients.