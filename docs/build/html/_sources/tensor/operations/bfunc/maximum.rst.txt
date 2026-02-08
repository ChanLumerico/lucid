lucid.maximum
=============

.. autofunction:: lucid.maximum

The `maximum` function computes the element-wise maximum of two `Tensor` objects. 
It returns a new `Tensor` containing the larger of the corresponding elements 
from `a` and `b`, with gradient support for backpropagation.

Function Signature
------------------

.. code-block:: python

    def maximum(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first tensor in the operation.
- **b** (*Tensor*): The second tensor in the operation.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is the maximum of the corresponding 
    elements from `a` and `b`. If either `a` or `b` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the element-wise maximum operation is:

.. math::

    \text{out}_i = \max(a_i, b_i)

where :math:`a_i` and :math:`b_i` are the :math:`i`-th elements of tensors `a` and `b`.

Backward Gradient Calculation
-----------------------------

For each tensor `a` and `b` involved in the `maximum` operation, the gradient 
with respect to the output (`out`) is computed as follows:

.. math::

    \frac{\partial \text{out}_i}{\partial a_i} = 
    \begin{cases} 
    1 & \text{if } a_i > b_i \\
    0 & \text{otherwise}
    \end{cases}, \quad 
    \frac{\partial \text{out}_i}{\partial b_i} = 
    \begin{cases} 
    1 & \text{if } b_i > a_i \\
    0 & \text{otherwise}
    \end{cases}

Examples
--------

Using `maximum` to compute the element-wise maximum:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([3.0, 5.0, 2.0], requires_grad=True)
    >>> b = Tensor([4.0, 2.0, 3.0], requires_grad=True)
    >>> out = lucid.maximum(a, b)
    >>> print(out)
    Tensor([4.0, 5.0, 3.0], grad=None)

After calling `backward()` on `out`, gradients for `a` and `b` will be 
accumulated based on the backpropagation rules:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [0.0, 1.0, 0.0]
    >>> print(b.grad)
    [1.0, 0.0, 1.0]
