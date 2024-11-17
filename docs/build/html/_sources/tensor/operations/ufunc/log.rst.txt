lucid.log
=========

.. autofunction:: lucid.log

The `log` function computes the natural logarithm (base e) 
of each element in the input tensor, similar to NumPy’s `np.log`.

Function Signature
------------------

.. code-block:: python

    def log(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the natural logarithm is computed.

Returns
-------

- **Tensor**: 
    A new tensor containing the element-wise natural logarithm of the input tensor. 
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for `log` is:

.. math::

    \mathbf{out}_i = \log(\mathbf{a}_i)

where :math:`\mathbf{a}_i` is the element of the input tensor **a**, 
and :math:`\mathbf{out}_i` is the corresponding element of the output tensor.

Backward Gradient Calculation
-----------------------------

For a tensor **a** involved in the `log` operation, 
the gradient with respect to the output (**out**) is computed as:

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = \frac{1}{\mathbf{a}_i}

This means that for each element in the input tensor, 
the gradient is the reciprocal of the corresponding value in the tensor.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 2, 3], requires_grad=True)
    >>> out = lucid.log(a)
    >>> print(out)
    Tensor([0.         0.69314718 1.09861229], grad=None)

The `log` function supports tensors of arbitrary shape:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> out = lucid.log(a)
    >>> print(out)
    Tensor([[0.         0.69314718] [1.09861229 1.38629436]], grad=None)
