lucid.log2
==========

.. autofunction:: lucid.log2

The `log2` function computes the base-2 logarithm of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def log2(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the base-2 logarithm is computed.

Returns
-------

- **Tensor**:  
  A new tensor containing the element-wise base-2 logarithm of the input tensor.  
  If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for `log2` is:

.. math::

    \mathbf{out}_i = \log_2(\mathbf{a}_i)

where :math:`\mathbf{a}_i` is the element of the input tensor **a**,  
and :math:`\mathbf{out}_i` is the corresponding element of the output tensor.

Backward Gradient Calculation
-----------------------------

For a tensor **a** involved in the `log2` operation,  
the gradient with respect to the output (**out**) is computed as:

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = \frac{1}{\mathbf{a}_i \log(2)}

This means that for each element in the input tensor,  
the gradient is the reciprocal of the corresponding value multiplied by :math:`\log(2)`.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 2, 3], requires_grad=True)
    >>> out = lucid.log2(a)
    >>> print(out)
    Tensor([0.         1.         1.5849625], grad=None)

The `log2` function supports tensors of arbitrary shape:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.log2(a)
    >>> print(out)
    Tensor([[0.        1.       ] 
            [1.5849625 2.       ]], grad=None)
