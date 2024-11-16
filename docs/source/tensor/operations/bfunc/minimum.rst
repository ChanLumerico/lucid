lucid.minimum
=============

.. autofunction:: lucid.minimum

The `minimum` function computes the element-wise minimum of two `Tensor` objects. 
It returns a new `Tensor` containing the smaller of the corresponding elements 
from `a` and `b`, with gradient support for backpropagation.

Function Signature
------------------

.. code-block:: python

    def minimum(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first tensor in the operation.
- **b** (*Tensor*): The second tensor in the operation.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is the minimum of the corresponding elements 
    from `a` and `b`. If either `a` or `b` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the element-wise minimum operation is:

.. math::

    \text{out}_i = \min(a_i, b_i)

where :math:`a_i` and :math:`b_i` are the :math:`i`-th elements of tensors `a` and `b`.

Backward Gradient Calculation
-----------------------------

For each tensor `a` and `b` involved in the `minimum` operation, 
the gradient with respect to the output (`out`) is computed as follows:

.. math::

    \frac{\partial \text{out}_i}{\partial a_i} = 
    \begin{cases} 
    1 & \text{if } a_i < b_i \\
    0 & \text{otherwise}
    \end{cases}, \quad 
    \frac{\partial \text{out}_i}{\partial b_i} = 
    \begin{cases} 
    1 & \text{if } b_i < a_i \\
    0 & \text{otherwise}
    \end{cases}

Examples
--------

Using `minimum` to compute the element-wise minimum:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([3.0, 5.0, 2.0], requires_grad=True)
    >>> b = Tensor([4.0, 2.0, 3.0], requires_grad=True)
    >>> out = lucid.minimum(a, b)
    >>> print(out)
    Tensor([3.0, 2.0, 2.0], grad=None)

After calling `backward()` on `out`, gradients for `a` and `b` will be 
accumulated based on the backpropagation rules:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [1.0, 0.0, 1.0]
    >>> print(b.grad)
    [0.0, 1.0, 0.0]
