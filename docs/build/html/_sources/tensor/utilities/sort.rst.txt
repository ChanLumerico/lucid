lucid.sort
==========

.. autofunction:: lucid.sort

The ``sort`` function sorts the elements of a tensor along a given axis and returns 
both the sorted values and the indices that would sort the input. This operation supports 
both ascending and descending order.

Function Signature
------------------

.. code-block:: python

    def sort(input_: Tensor, axis: int = -1, descending: bool = False) -> tuple[Tensor, Tensor]:

Parameters
----------

- **input_** (*Tensor*):  
  The input tensor to be sorted.

- **axis** (*int*, optional):  
  The axis along which to sort. Default is `-1` (last axis).

- **descending** (*bool*, optional):  
  Whether to sort in descending order. Default is `False`.

Returns
-------

- **values** (*Tensor*):  
  A tensor of the same shape as `input_` with elements sorted along the specified axis.

- **indices** (*Tensor*):  
  A tensor of indices that map each sorted element to its position in the original input.

Gradient Computation
--------------------

Gradients are propagated by reversing the sort using the indices returned.
This ensures gradients are distributed back to the correct locations in the original tensor.

Example
-------

Sorting a tensor row-wise and computing gradients:

.. code-block:: python

    >>> import lucid
    >>> x = lucid.Tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], requires_grad=True)
    >>> values, indices = x.sort(axis=1, descending=False)
    >>> loss = values.sum()
    >>> loss.backward()
    >>> print(values.data)
    [[1. 2. 3.]
     [4. 5. 6.]]
    >>> print(indices.data)
    [[1 2 0]
     [1 2 0]]
    >>> print(x.grad)
    [[1. 1. 1.]
     [1. 1. 1.]]

.. note::

    - The returned indices can be used to reverse or reconstruct the original order.
    - If `descending=True`, the result is reversed after sorting.
    - This function fully supports autograd and FLOPs tracking.
