lucid.cos
=========

.. autofunction:: lucid.cos

The `cos` function computes the element-wise cosine of each element in the input tensor.

Function Signature
------------------

.. code-block:: python

    def cos(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor for which the cosine is computed.

Returns
-------

- **Tensor**:  
    A new tensor containing the element-wise cosine of the input tensor.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

.. math::

    \mathbf{out}_i = \cos(\mathbf{a}_i)

Backward Gradient Calculation
-----------------------------

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = -\sin(\mathbf{a}_i)

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([0, math.pi / 2, math.pi], requires_grad=True)
    >>> out = lucid.cos(a)
    >>> print(out)
    Tensor([1. 0. -1.], grad=None)
