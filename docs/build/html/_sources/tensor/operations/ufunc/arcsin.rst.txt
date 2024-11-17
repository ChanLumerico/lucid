lucid.arcsin
============

.. autofunction:: lucid.arcsin

The `arcsin` function computes the element-wise inverse sine of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def arcsin(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the inverse sine is computed.  
  Values should be in the range :math:`[-1, 1]`.

Returns
-------

- **Tensor**:  
    A new tensor containing the element-wise inverse sine of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = \arcsin(\mathbf{a}_i)

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = \frac{1}{\sqrt{1 - \mathbf{a}_i^2}}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([0, 0.5, 1], requires_grad=True)
    >>> out = lucid.arcsin(a)
    >>> print(out)
    Tensor([0. 0.52359878 1.57079633], grad=None)
