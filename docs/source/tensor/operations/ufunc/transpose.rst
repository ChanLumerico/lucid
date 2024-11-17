lucid.transpose
===============

The `transpose` function performs a permutation of the dimensions of the input tensor.
If the axes are not specified, it reverses the dimensions.

Function Signature
------------------

.. code-block:: python

    def transpose(a: Tensor, axes: list[int] | None = None) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor to transpose.
- **axes** (*list[int] | None*, optional): 
    A list of integers representing the desired order of the axes. 
    If `None`, the dimensions are reversed. Defaults to `None`.

Returns
-------

- **Tensor**: 
    A new tensor with the dimensions permuted according to 
    the specified axes or reversed if `axes` is `None`.

Forward Calculation
-------------------

The forward calculation for `transpose` is based on the provided `axes`:

.. math::

    \mathbf{out}_i = \mathbf{a}_{\sigma(i)}

where :math:`\mathbf{a}` is the input tensor and :math:`\sigma(i)` denotes the permutation of 
the dimensions defined by the `axes` list. If `axes` is not specified, the dimensions are reversed.

Backward Gradient Calculation
-----------------------------

For the backward pass, the gradient with respect to the input tensor **a** is computed as 
the transpose of the gradient with respect to the output tensor **out**:

.. math::

    \frac{\partial \mathbf{a}_i}{\partial \mathbf{out}_j} = \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_j}

This means that the gradient with respect to the input tensor is 
the transposed gradient of the output tensor.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.transpose(a)
    >>> print(out)
    Tensor([[1 3] [2 4]], grad=None)

The `transpose` function supports tensors of arbitrary shape:

.. code-block:: python

    >>> a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    >>> out = lucid.transpose(a, axes=[2, 1, 0])
    >>> print(out)
    Tensor([[[1 5] [3 7]] [[2 6] [4 8]]], grad=None)

.. note::

    - The `axes` argument allows you to specify the exact order of the dimensions. 
      If not provided, the function simply reverses the dimensions.

    - The `transpose` function supports tensors with any number of dimensions, 
      as long as the specified axes are valid for the shape of the tensor.

