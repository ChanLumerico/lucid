lucid.mean
==========

.. autofunction:: lucid.mean

The `mean` function computes the mean (average) of the elements in 
the input tensor along the specified axis or axes.

Function Signature
------------------

.. code-block:: python

    def mean(
        a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
    ) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor to compute the mean from.

- **axis** (*int | tuple[int] | None*, optional): 
    The axis or axes along which to compute the mean. 
    If `None`, the mean is computed over all elements. Defaults to `None`.

- **keepdims** (*bool*, optional): 
    Whether to retain the reduced dimensions in the output tensor. 
    If `True`, the reduced dimensions will be preserved with size 1. 
    Defaults to `False`.

Returns
-------

- **Tensor**: 
    A tensor containing the mean of the input tensor's elements along the specified axes. 
    If `keepdims` is `True`, the reduced dimensions will be kept with size 1.

Forward Calculation
-------------------

The forward calculation for `mean` is:

.. math::

    \mathbf{out}_i = \frac{1}{N} \sum_{\mathbf{a}_j \in \text{axis}} \mathbf{a}_j

where :math:`\mathbf{a}_j` represents the elements along the specified axis, 
and :math:`N` is the number of elements in the axis. 
If `axis` is `None`, the mean is computed over all elements of the tensor.

Backward Gradient Calculation
-----------------------------

For the backward pass, the gradient with respect to the input tensor **a** 
is computed by distributing the gradient of the output tensor across the elements:

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_j} = \frac{1}{N}

This means that the gradient with respect to each element is the same, 
and the gradients are averaged along the specified axis during the backward pass.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.mean(a)
    >>> print(out)
    Tensor(2.5, grad=None)

The `mean` function supports tensors of arbitrary shape and axis combinations:

.. code-block:: python

    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.mean(a, axis=0)
    >>> print(out)
    Tensor([2. 3.], grad=None)

    >>> a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    >>> out = lucid.mean(a, axis=(0, 2), keepdims=True)
    >>> print(out)
    Tensor([[[4.]]] grad=None)

.. note::

    - The `axis` parameter can be an integer or a tuple of integers for multi-dimensional means.

    - The `keepdims` parameter ensures that the output tensor retains the reduced dimensions, 
      which can be useful for broadcasting during further operations.
