lucid.sign
==========

.. autofunction:: lucid.sign

The `sign` function computes the element-wise sign of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def sign(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*):  
  The input tensor whose signs are computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the sign values of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i =
    \begin{cases} 
      -1 & \text{if } \mathbf{a}_i < 0 \\
       0 & \text{if } \mathbf{a}_i = 0 \\
       1 & \text{if } \mathbf{a}_i > 0
    \end{cases}

Backward Gradient Calculation
-----------------------------

The gradient of `sign` is zero everywhere except at discontinuities.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([-1, -0.5, 0, 0.5, 1], requires_grad=True)
    >>> out = lucid.sign(a)
    >>> print(out)
    Tensor([-1. -1.  0.  1.  1.], grad=None)
