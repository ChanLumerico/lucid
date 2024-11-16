lucid.div
=========

.. autofunction:: lucid.div

The `div` function performs element-wise division between two `Tensor` objects. 
It returns a new `Tensor` representing the quotient, with gradient 
support for backpropagation.

Function Signature
------------------

.. code-block:: python

    def div(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The numerator tensor in the division operation.
- **b** (*Tensor*): The denominator tensor in the division operation.

Returns
-------

- **Tensor**: 
    A new `Tensor` representing the element-wise quotient of `a` divided by `b`. 
    If either `a` or `b` requires gradients, the resulting tensor 
    will also require gradients.

Forward Calculation
-------------------

The forward calculation for the division operation is:

.. math::

    \text{out} = \frac{a}{b}

where :math:`a` and :math:`b` are the data contained in the tensors `a` and `b`, respectively.

Backward Gradient Calculation
-----------------------------

For each tensor `a` and `b` involved in the division, 
the gradient with respect to the output (`out`) is computed as follows:

.. math::

    \frac{\partial \text{out}}{\partial a} = \frac{1}{b}, \quad 
    \frac{\partial \text{out}}{\partial b} = -\frac{a}{b^2}

Examples
--------

Using `div` to divide two tensors:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([6.0, 8.0, 10.0], requires_grad=True)
    >>> b = Tensor([2.0, 4.0, 5.0], requires_grad=True)
    >>> out = lucid.div(a, b)
    >>> print(out)
    Tensor([3.0, 2.0, 2.0], grad=None)

After calling `backward()` on `out`, gradients for `a` and `b` will be 
accumulated based on the backpropagation rules:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [0.5, 0.25, 0.2]
    >>> print(b.grad)
    [-1.5, -0.5, -0.4]
