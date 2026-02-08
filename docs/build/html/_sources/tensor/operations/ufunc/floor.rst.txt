lucid.floor
===========

.. autofunction:: lucid.floor

The `floor` function returns a new tensor with the element-wise floor of the input tensor.  
Each value is rounded down to the nearest integer less than or equal to the original value.

Function Signature
------------------

.. code-block:: python

    def floor(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*):  
  The input tensor to apply the floor function to.

Returns
-------

- **Tensor**:  
  A new tensor where each element is the floor of the corresponding input value.  
  The output tensor has the same shape and device as the input.

Mathematical Expression
-----------------------

.. math::

    \text{floor}(x_i) = \lfloor x_i \rfloor

.. warning::

   This function is **non-differentiable**, so its gradient is always zero.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.Tensor([2.9, -1.3, 0.0])
    >>> lucid.floor(x)
    Tensor([2.0, -2.0, 0.0], grad=None)

.. tip::

    Use `floor` to discretize continuous values — particularly useful in 
    geometric and index-based tensor logic. However, avoid using it during 
    training due to its non-differentiable nature.
