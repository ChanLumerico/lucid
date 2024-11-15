lucid.sub
=========

.. autofunction:: lucid.sub

The `sub` function performs element-wise subtraction between two `Tensor` objects. 
It returns a new `Tensor` representing the difference, with gradient support 
for backpropagation.

Function Signature
------------------

.. code-block:: python

    def sub(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first tensor in the subtraction operation.
- **b** (*Tensor*): The second tensor in the subtraction operation.

Returns
-------

- **Tensor**: 
    A new `Tensor` representing the element-wise difference between 
    `a` and `b`. If either `a` or `b` requires gradients, the resulting tensor will 
    also require gradients.

Forward Calculation
-------------------

The forward calculation for the subtraction operation is:

.. math::

    \text{out} = a - b

where :math:`a` and :math:`b` are the data contained in the tensors 
`a` and `b`, respectively.

Backward Gradient Calculation
-----------------------------

For each tensor `a` and `b` involved in the subtraction, 
the gradient with respect to the output (`out`) is computed as follows:

.. math::

    \frac{\partial \text{out}}{\partial a} = 1, 
    \quad \frac{\partial \text{out}}{\partial b} = -1

Examples
--------

Using `sub` to subtract one tensor from another:

.. code-block:: python

    >>> a = Tensor([5.0, 7.0, 9.0], requires_grad=True)
    >>> b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    >>> out = sub(a, b)
    >>> print(out)
    Tensor([1.0, 2.0, 3.0], grad=None)

After calling `backward()` on `out`, gradients for `a` and `b` will be accumulated 
based on the backpropagation rules:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [1.0, 1.0, 1.0]
    >>> print(b.grad)
    [-1.0, -1.0, -1.0]
