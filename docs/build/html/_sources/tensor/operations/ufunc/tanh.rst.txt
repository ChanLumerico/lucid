lucid.tanh
==========

.. autofunction:: lucid.tanh

The `tanh` function computes the element-wise hyperbolic tangent of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def tanh(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the hyperbolic tangent is computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the element-wise hyperbolic tangent of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = \tanh(\mathbf{a}_i) = \frac{\sinh(\mathbf{a}_i)}{\cosh(\mathbf{a}_i)} = 
    \frac{e^{\mathbf{a}_i} - e^{-\mathbf{a}_i}}{e^{\mathbf{a}_i} + e^{-\mathbf{a}_i}}

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = 1 - \tanh^2(\mathbf{a}_i)

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([0, 1, 2], requires_grad=True)
    >>> out = lucid.tanh(a)
    >>> print(out)
    Tensor([0. 0.76159416 0.96402758], grad=None)
