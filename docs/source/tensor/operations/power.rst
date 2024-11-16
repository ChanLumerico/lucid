lucid.power
===========

.. autofunction:: lucid.power

The `power` function raises each element of the first tensor `a` to the power 
specified by the corresponding element in the second tensor `b`. 
It supports gradient computation for backpropagation.

Function Signature
------------------

.. code-block:: python

    def power(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The base tensor.
- **b** (*Tensor*): The exponent tensor.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is computed as the corresponding 
    element of `a` raised to the power of the corresponding element of `b`. 
    If either `a` or `b` requires gradients, the resulting tensor will also 
    require gradients.

Forward Calculation
-------------------

The forward calculation for the `power` operation is:

.. math::

    \text{out}_i = a_i^{b_i}

where :math:`a_i` and :math:`b_i` are the :math:`i`-th elements of tensors `a` and `b`.

Backward Gradient Calculation
-----------------------------

For tensors `a` and `b` involved in the `power` operation, the gradients 
with respect to the output (`out`) are computed as follows:

**Gradient with respect to** :math:`a`:

.. math::

    \frac{\partial \text{out}_i}{\partial a_i} = b_i \cdot a_i^{b_i - 1}

**Gradient with respect to** :math:`b`:

.. math::

    \frac{\partial \text{out}_i}{\partial b_i} = a_i^{b_i} \cdot \ln(a_i)

.. note::

    - The gradient with respect to `b` is undefined for :math:`a_i \leq 0`, 
    as the natural logarithm function :math:`\ln(a_i)` is not defined for 
    non-positive numbers.

    - Ensure that all elements of `a` are positive when `b` requires gradients.

Examples
--------

Using `power` to compute the element-wise power of two tensors:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    >>> b = Tensor([3.0, 2.0, 1.0], requires_grad=True)
    >>> out = lucid.power(a, b)
    >>> print(out)
    Tensor([8.0, 9.0, 4.0], grad=None)

After calling `backward()` on `out`, gradients for `a` and `b` 
are computed based on the backpropagation rules:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [12.0, 6.0, 1.0]  # Corresponding to b * a^(b-1)
    >>> print(b.grad)
    [5.545, 9.887, 2.772]  # Corresponding to a^b * ln(a)
