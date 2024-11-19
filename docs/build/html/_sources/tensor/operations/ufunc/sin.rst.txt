lucid.sin
=========

.. autofunction:: lucid.sin

The `sin` function computes the element-wise sine of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def sin(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the sine is computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the element-wise sine of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for `sin` is:

.. math::

    \mathbf{out}_i = \sin(\mathbf{a}_i)

where :math:`\mathbf{a}_i` is the element of the input tensor **a**,  
and :math:`\mathbf{out}_i` is the corresponding element of the output tensor.

Backward Gradient Calculation
-----------------------------

For a tensor **a** involved in the `sin` operation,  
the gradient with respect to the output (**out**) is computed as:

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = \cos(\mathbf{a}_i)

This means that for each element in the input tensor,  
the gradient is the cosine of the corresponding value in the tensor.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([0, math.pi / 2, math.pi], requires_grad=True)
    >>> out = lucid.sin(a)
    >>> print(out)
    Tensor([0.         1.         0.        ], grad=None)

The `sin` function supports tensors of arbitrary shape:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[0, math.pi / 4], [math.pi / 2, math.pi]], requires_grad=True)
    >>> out = lucid.sin(a)
    >>> print(out)
    Tensor([[0.         0.70710678] [1.         0.        ]], grad=None)
