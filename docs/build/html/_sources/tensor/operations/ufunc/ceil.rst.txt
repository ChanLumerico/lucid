lucid.ceil
==========

.. autofunction:: lucid.ceil

The `ceil` function returns a new tensor with the element-wise ceiling of the input tensor.  
Each value is rounded up to the smallest integer greater than or equal to the original value.

Function Signature
------------------

.. code-block:: python

    def ceil(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*):  
  The input tensor to apply the ceiling function to.

Returns
-------

- **Tensor**:  
  A new tensor where each element is the ceiling of the corresponding input value.  
  The output tensor has the same shape and device as the input.

Mathematical Expression
-----------------------

.. math::

    \text{ceil}(x_i) = \lceil x_i \rceil

.. warning::

   This function is **non-differentiable**, so its gradient is always zero.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.Tensor([2.1, -1.8, 0.0])
    >>> lucid.ceil(x)
    Tensor([3.0, -1.0, 0.0], grad=None)

.. tip::

    Use `ceil` to ensure upper-bound rounding — useful in spatial grid applications 
    and bucket partitioning. Avoid using it during model training as it disrupts 
    gradient flow.
