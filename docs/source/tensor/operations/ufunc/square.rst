lucid.square
============

.. autofunction:: lucid.square

The `square` function computes the element-wise square of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def square(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*):  
  The input tensor to be squared element-wise.

Returns
-------

- **Tensor**:  
    A new tensor containing the squared values of the elements in the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = (\mathbf{a}_i)^2

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = 2 \mathbf{a}_i

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 2, 3], requires_grad=True)
    >>> out = lucid.square(a)
    >>> print(out)
    Tensor([1. 4. 9.], grad=None)
