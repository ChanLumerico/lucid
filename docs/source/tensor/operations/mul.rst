lucid.mul
=========

.. autofunction:: lucid.mul

The `mul` function performs element-wise multiplication between two `Tensor` objects. 
It returns a new `Tensor` representing the product, with gradient support for backpropagation.

Function Signature
------------------

.. code-block:: python

    def mul(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first tensor in the multiplication operation.
- **b** (*Tensor*): The second tensor in the multiplication operation.

Returns
-------

- **Tensor**: 
    A new `Tensor` representing the element-wise product of `a` and `b`. 
    If either `a` or `b` requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the multiplication operation is:

.. math::

    \text{out} = a \cdot b

where :math:`a` and :math:`b` are the data contained in the tensors `a` and `b`, respectively.

Backward Gradient Calculation
-----------------------------

For each tensor `a` and `b` involved in the multiplication, 
the gradient with respect to the output (`out`) is computed as follows:

.. math::

    \frac{\partial \text{out}}{\partial a} = b, 
    \quad \frac{\partial \text{out}}{\partial b} = a

Examples
--------

Using `mul` to multiply two tensors:

.. code-block:: python

    >>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    >>> out = mul(a, b)
    >>> print(out)
    Tensor([4.0, 10.0, 18.0], grad=None)

After calling `backward()` on `out`, gradients for `a` and `b` 
will be accumulated based on the backpropagation rules:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [4.0, 5.0, 6.0]
    >>> print(b.grad)
    [1.0, 2.0, 3.0]
