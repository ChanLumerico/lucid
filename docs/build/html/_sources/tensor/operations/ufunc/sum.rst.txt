lucid.sum
=========

.. autofunction:: lucid.sum

The `sum` function computes the sum of the elements in the input tensor along the specified axis or axes.

Function Signature
------------------

.. code-block:: python

    def sum(
        a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
    ) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor to compute the sum from.

- **axis** (*int | tuple[int] | None*, optional): 
    The axis or axes along which to compute the sum. 
    If `None`, the sum is computed over all elements. Defaults to `None`.

- **keepdims** (*bool*, optional): 
    Whether to retain the reduced dimensions in the output tensor. 
    If `True`, the reduced dimensions will be preserved with size 1. Defaults to `False`.

Returns
-------

- **Tensor**: 
    A tensor containing the sum of the input tensor's elements along the specified axes. 
    If `keepdims` is `True`, the reduced dimensions will be kept with size 1.

Forward Calculation
-------------------

The forward calculation for `sum` is:

.. math::

    \mathbf{out}_i = \sum_{\mathbf{a}_j \in \text{axis}} \mathbf{a}_j

Where :math:`\mathbf{a}_j` are the elements along the specified axis, 
and the sum operation is performed over the selected axis or axes.

If `axis` is `None`, the sum is computed over all elements of the tensor.

Backward Gradient Calculation
-----------------------------

For the backward pass, the gradient with respect to the input tensor **a** 
is computed by distributing the gradient of the output tensor along the specified axis:

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_j} = 1

This means the gradient with respect to each element in the input tensor is 1, 
and the gradients are summed along the specified axis during the backward pass.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.sum(a)
    >>> print(out)
    Tensor(10, grad=None)

The `sum` function supports tensors of arbitrary shape and axis combinations:

.. code-block:: python

    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.sum(a, axis=0)
    >>> print(out)
    Tensor([4 6], grad=None)

    >>> a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    >>> out = lucid.sum(a, axis=(0, 2), keepdims=True)
    >>> print(out)
    Tensor([[[ 6]] [ 8]], grad=None)

.. note::

    - The `axis` parameter can be an integer or a tuple of integers for multi-dimensional sums.

    - The `keepdims` parameter ensures that the output tensor retains the reduced dimensions, 
      which can be useful for broadcasting during further operations.

