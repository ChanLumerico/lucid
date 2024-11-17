lucid.var
=========

The `var` function computes the variance of the elements in 
the input tensor along the specified axis or axes.

Function Signature
------------------

.. code-block:: python

    def var(
        a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
    ) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor to compute the variance from.

- **axis** (*int | tuple[int] | None*, optional): 
    The axis or axes along which to compute the variance. If `None`, 
    the variance is computed over all elements. Defaults to `None`.

- **keepdims** (*bool*, optional): 
    Whether to retain the reduced dimensions in the output tensor. 
    If `True`, the reduced dimensions will be preserved with size 1. Defaults to `False`.

Returns
-------

- **Tensor**: 
    A tensor containing the variance of the input tensor's elements along the specified axes. 
    If `keepdims` is `True`, the reduced dimensions will be kept with size 1.

Forward Calculation
-------------------

The forward calculation for `var` is:

.. math::

    \mathbf{out}_i = \frac{1}{N} \sum_{\mathbf{a}_j \in \text{axis}} (\mathbf{a}_j - \mu)^2

where :math:`\mathbf{a}_j` represents the elements along the specified axis, 
:math:`\mu` is the mean of the elements in the axis, and N is the number of elements in the axis. 

If `axis` is `None`, the variance is computed over all elements of the tensor.

Backward Gradient Calculation
-----------------------------

For the backward pass, the gradient with respect to the input tensor **a** is 
computed by distributing the gradient of the output tensor across the elements:

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_j} = \frac{2(\mathbf{a}_j - \mu)}{N}

This means that the gradient with respect to each element is proportional 
to the difference between the element and the mean, scaled by :math:`2/N`.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.var(a)
    >>> print(out)
    Tensor(1.25, grad=None)

The `var` function supports tensors of arbitrary shape and axis combinations:

.. code-block:: python

    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.var(a, axis=0)
    >>> print(out)
    Tensor([1. 1.], grad=None)

    >>> a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    >>> out = lucid.var(a, axis=(0, 2), keepdims=True)
    >>> print(out)
    Tensor([[[2.]]] grad=None)

.. note::

    - The `axis` parameter can be an integer or a tuple of integers for multi-dimensional variances.

    - The `keepdims` parameter ensures that the output tensor retains the reduced dimensions, 
      which can be useful for broadcasting during further operations.

    - Variance is calculated using the formula :math:`\frac{1}{N} \sum (\mathbf{a}_j - \mu)^2`, 
      where :math:`\mu` is the mean of the elements along the specified axis.
