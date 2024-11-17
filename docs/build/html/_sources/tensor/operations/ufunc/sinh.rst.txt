lucid.sinh
==========

.. autofunction:: lucid.sinh

The `sinh` function computes the element-wise hyperbolic sine of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def sinh(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the hyperbolic sine is computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the element-wise hyperbolic sine of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = \sinh(\mathbf{a}_i) = \frac{e^{\mathbf{a}_i} - e^{-\mathbf{a}_i}}{2}

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = \cosh(\mathbf{a}_i)

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([0, 1, 2], requires_grad=True)
    >>> out = lucid.sinh(a)
    >>> print(out)
    Tensor([0. 1.17520119 3.62686041], grad=None)
