lucid.swapaxes
==============

.. autofunction:: lucid.swapaxes

The `swapaxes` function swaps two specified axes of a tensor, similar to NumPy's 
`np.swapaxes`. This allows for reordering dimensions while preserving the underlying data.

Function Signature
------------------

.. code-block:: python

    def swapaxes(a: Tensor, axis1: int, axis2: int) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor whose axes will be swapped.
- **axis1** (*int*): The first axis to swap.
- **axis2** (*int*): The second axis to swap.

Returns
-------

- **Tensor**:
  A new `Tensor` with `axis1` and `axis2` swapped. If `a` requires gradients, 
  the resulting tensor will also require gradients.

Axis Swapping Rules
-------------------

- `swapaxes(a, 0, 1)` swaps the first and second axes, useful in reshaping operations.
- For higher-dimensional tensors, swapping axes can help in 
  aligning data for operations such as matrix multiplication or convolutions.

Backward Gradient Calculation
-----------------------------

Since `swapaxes` only reorders dimensions, the gradient computation follows 
the same transformation:

.. math::

    \frac{\partial L}{\partial \mathbf{a}} = 
    \text{swapaxes}\left(\frac{\partial L}{\partial \text{output}}, \text{axis1}, \text{axis2}\right)

Examples
--------

**Swapping two axes in a 2D tensor**

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    >>> swapped = lucid.swapaxes(a, 0, 1)
    >>> print(swapped)
    Tensor([[1, 4],
            [2, 5],
            [3, 6]], grad=None)

**Preserving gradient tracking**

.. code-block:: python

    >>> swapped.backward()
    >>> print(a.grad)
    [[1, 1, 1],
     [1, 1, 1]]  # Gradients are swapped accordingly

**Using `swapaxes` for batch-first transformations**

.. code-block:: python

    >>> batch_tensor = lucid.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> new_tensor = lucid.swapaxes(batch_tensor, 0, 1)
    >>> print(new_tensor.shape)
    (2, 2, 2)

