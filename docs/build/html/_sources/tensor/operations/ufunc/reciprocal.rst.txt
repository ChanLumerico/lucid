lucid.reciprocal
================

.. autofunction:: lucid.reciprocal

The `reciprocal` function computes the element-wise reciprocal of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def reciprocal(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*):  
  The input tensor for which reciprocals are computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the reciprocals of the elements in the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = \frac{1}{\mathbf{a}_i}

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = -\frac{1}{\mathbf{a}_i^2}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 2, 4], requires_grad=True)
    >>> out = lucid.reciprocal(a)
    >>> print(out)
    Tensor([1. 0.5 0.25], grad=None)
