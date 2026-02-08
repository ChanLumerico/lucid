lucid.cube
==========

.. autofunction:: lucid.cube

The `cube` function computes the element-wise cube of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def cube(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*):  
  The input tensor to be cubed element-wise.

Returns
-------

- **Tensor**:  
    A new tensor containing the cubed values of the elements in the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = (\mathbf{a}_i)^3

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = 3 (\mathbf{a}_i)^2

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 2, 3], requires_grad=True)
    >>> out = lucid.cube(a)
    >>> print(out)
    Tensor([1. 8. 27.], grad=None)
