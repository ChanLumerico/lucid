lucid.topk
==========

.. autofunction:: lucid.topk

The `topk` function returns the top `k` largest or smallest values along a given axis, 
along with their indices.

Function Signature
------------------

.. code-block:: python

    def topk(
        input_: Tensor, 
        k: int, 
        axis: int = -1, 
        largest: bool = True, 
        sorted: bool = True
    ) -> tuple[Tensor, Tensor]:

Parameters
----------
- **input_** (*Tensor*):
  The input tensor to extract values from.

- **k** (*int*):
  The number of top elements to return.

- **axis** (*int*, optional):
  The axis along which to retrieve the top values. Default is `-1`.

- **largest** (*bool*, optional):
  If `True` (default), returns the `k` largest elements. 
  Otherwise, returns the `k` smallest.

- **sorted** (*bool*, optional):
  If `True` (default), the resulting `k` values will be sorted in descending 
  (or ascending) order depending on `largest`.

Returns
-------
- **tuple[Tensor, Tensor]**:
  A tuple of two tensors:
  
  - The first contains the top `k` values.
  - The second contains the indices of those values in the original tensor.

Gradient Computation
--------------------
Gradients are propagated only to the selected top `k` positions. 
Other positions receive a gradient of zero.

.. math::

    \frac{\partial \text{topk}(x)}{\partial x_i} = 
    \begin{cases}
        \text{grad}_i & \text{if } i \in \text{topk_indices} \\
        0 & \text{otherwise}
    \end{cases}

Example
-------

Finding the top-2 largest values along `axis=1`:

.. code-block:: python

    >>> import lucid
    >>> x = lucid.Tensor([[10., 5., 8., 2.], [1., 4., 3., 9.]], requires_grad=True)
    >>> values, indices = lucid.topk(x, k=2, axis=1, largest=True, sorted=True)
    >>> print(values)
    [[10.  8.]
     [ 9.  4.]]
    >>> print(indices)
    [[0 2]
     [3 1]]

    >>> loss = values.sum()
    >>> loss.backward()
    >>> print(x.grad)
    [[1. 0. 1. 0.]
     [0. 1. 0. 1.]]

.. note::

    `topk` ensures proper gradient routing only through the selected indices.

