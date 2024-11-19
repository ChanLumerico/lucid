lucid.tan
=========

.. autofunction:: lucid.tan

The `tan` function computes the element-wise tangent of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def tan(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the tangent is computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the element-wise tangent of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = \tan(\mathbf{a}_i)

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = 1 + \tan^2(\mathbf{a}_i)

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([0, math.pi / 4, math.pi / 2], requires_grad=True)
    >>> out = lucid.tan(a)
    >>> print(out)
    Tensor([0. 1. inf], grad=None)
