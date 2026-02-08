lucid.trace
===========

.. autofunction:: lucid.trace

The `trace` function computes the sum of the diagonal elements of a tensor. 
If the tensor is 2D, it behaves similarly to `numpy.trace`, returning the sum of the diagonal elements.

Function Signature
------------------

.. code-block:: python

    def trace(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): 
    The input tensor for which the trace is calculated. The tensor should be at least 2-dimensional.

Returns
-------

- **Tensor**: 
    A new tensor containing the sum of the diagonal elements of the input tensor. 
    If the input tensor has more than two dimensions, only the diagonal elements 
    of the 2D slices will be summed.

Forward Calculation
-------------------

The forward calculation for `trace` is:

.. math::

    \mathbf{out}_i = \sum_{i=1}^{n} \mathbf{a}_{ii}

where :math:`\mathbf{a}_{ii}` represents the diagonal elements of the input tensor 
:math:`\mathbf{a}`, and the sum is taken over the diagonal elements.

If the tensor has more than two dimensions, the trace is calculated over the 2D slices, 
i.e., each slice's diagonal elements are summed.

Backward Gradient Calculation
-----------------------------

For the backward pass, the gradient with respect to the input tensor **a** 
is computed by distributing the gradient of the output tensor along the diagonal elements:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}_{ii}} = 1

This means that the gradient with respect to each diagonal element is 1, 
and other elements do not contribute to the gradient.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.trace(a)
    >>> print(out)
    Tensor(5, grad=None)

The `trace` function supports tensors with more than two dimensions by 
summing the diagonal elements of each 2D slice:

.. code-block:: python

    >>> a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    >>> out = lucid.trace(a)
    >>> print(out)
    Tensor([5 15], grad=None)

.. note::

    - The input tensor must be at least 2-dimensional. 
      For 1D tensors, the result is simply the sum of all elements (equivalent to a sum operation).

    - The `trace` function operates element-wise on 2D slices of the tensor 
      for tensors with more than two dimensions.

