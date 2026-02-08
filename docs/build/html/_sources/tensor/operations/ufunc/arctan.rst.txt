lucid.arctan
============

.. autofunction:: lucid.arctan

The `arctan` function computes the element-wise inverse tangent of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def arctan(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the inverse tangent is computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the element-wise inverse tangent of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = \arctan(\mathbf{a}_i)

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = \frac{1}{1 + \mathbf{a}_i^2}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([0, 1, 10], requires_grad=True)
    >>> out = lucid.arctan(a)
    >>> print(out)
    Tensor([0. 0.78539816 1.47112767], grad=None)
