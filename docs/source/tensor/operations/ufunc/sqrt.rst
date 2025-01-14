lucid.sqrt
==========

.. autofunction:: lucid.sqrt

The `sqrt` function computes the element-wise square root of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def sqrt(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the square root is computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the element-wise square root of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for `sqrt` is:

.. math::

    \mathbf{out}_i = \sqrt{\mathbf{a}_i}

where :math:`\mathbf{a}_i` is the element of the input tensor **a**,  
and :math:`\mathbf{out}_i` is the corresponding element of the output tensor.

Backward Gradient Calculation
-----------------------------

For a tensor **a** involved in the `sqrt` operation,  
the gradient with respect to the output (**out**) is computed as:

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = \frac{1}{2 \sqrt{\mathbf{a}_i}}

This means that for each element in the input tensor,  
the gradient is half the reciprocal of the square root of the corresponding value in the tensor.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 4, 9], requires_grad=True)
    >>> out = lucid.sqrt(a)
    >>> print(out)
    Tensor([1.         2.         3.        ], grad=None)

The `sqrt` function supports tensors of arbitrary shape:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 4], [9, 16]], requires_grad=True)
    >>> out = lucid.sqrt(a)
    >>> print(out)
    Tensor([[1. 2.] [3. 4.]], grad=None)
