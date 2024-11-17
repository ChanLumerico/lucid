lucid.cosh
==========

.. autofunction:: lucid.cosh

The `cosh` function computes the element-wise hyperbolic cosine of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def cosh(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the hyperbolic cosine is computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the element-wise hyperbolic cosine of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = \cosh(\mathbf{a}_i) = \frac{e^{\mathbf{a}_i} + e^{-\mathbf{a}_i}}{2}

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = \sinh(\mathbf{a}_i)

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([0, 1, 2], requires_grad=True)
    >>> out = lucid.cosh(a)
    >>> print(out)
    Tensor([1. 1.54308063 3.76219569], grad=None)
