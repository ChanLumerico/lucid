lucid.add
=========

.. autofunction:: lucid.add

The `add` function performs element-wise addition between two `Tensor` objects. 
It returns a new `Tensor` as the sum of the inputs, with support for gradient 
calculation during backpropagation.

Function Signature
------------------

.. code-block:: python

    def add(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first tensor in the addition operation.

- **b** (*Tensor*): The second tensor in the addition operation.

Returns
-------

- **Tensor**: 
    A new `Tensor` representing the element-wise sum of `a` and `b`. 
    If either `a` or `b` requires gradients, the resulting tensor will also require 
    gradients.

Forward Calculation
-------------------

The forward calculation for the addition operation is straightforward:

.. math::

    \text{out} = a + b

where :math:`a` and :math:`b` are the data contained in the tensors `a` and `b`, 
respectively.

Backward Gradient Calculation
-----------------------------

For each tensor `a` and `b` involved in the addition, the gradient with respect 
to the output (`out`) is computed as:

.. math::

    \frac{\partial \text{out}}{\partial a} = 1, \quad \frac{\partial \text{out}}{\partial b} = 1

Thus, if `out` requires gradients, the gradients of `a` and `b` are simply the 
propagated gradient from `out` multiplied by 1:

- **Gradient of** :math:`a`: :math:`\text{a.grad} += \text{out.grad}`

- **Gradient of** :math:`b`: :math:`\text{b.grad} += \text{out.grad}`

Examples
--------

Using `add` to sum two tensors:

.. code-block:: python

    >>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    >>> out = add(a, b)
    >>> print(out)
    Tensor([5.0, 7.0, 9.0], grad=None)

After calling `backward()` on `out`, gradients for `a` and `b` will be 
accumulated based on the backpropagation rules:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [1.0, 1.0, 1.0]
    >>> print(b.grad)
    [1.0, 1.0, 1.0]
