lucid.exp
=========

.. autofunction:: lucid.exp

The `exp` function computes the exponential of each element in the input tensor, 
i.e., \(e^x\), where \(e\) is the base of natural logarithms.

Function Signature
------------------

.. code-block:: python

    def exp(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor whose elements will be exponentiated.

Returns
-------

- **Tensor**: 
    A new `Tensor` with the exponential of each element of `a`. 
    If `a` requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `exp` operation is:

.. math::

    \mathbf{out}_i = e^{\mathbf{a}_i}

where :math:`\mathbf{a}_i` are the elements of the input tensor **a**.

Backward Gradient Calculation
-----------------------------

For a tensor **a** involved in the `exp` operation, the gradient with respect to the output (**out**) is:

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} = e^{\mathbf{a}_i}

This means the gradient of **a** is the same as the value of the forward output.

Examples
--------

Using `exp` on a tensor:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([0.0, 1.0, 2.0], requires_grad=True)
    >>> out = lucid.exp(a)
    >>> print(out)
    Tensor([1.0, 2.7182818, 7.389056], grad=None)

Backpropagation computes gradients for `a`:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [1.0, 2.7182818, 7.389056]  # Corresponding to the exponential values of a

Using `exp` on a higher-dimensional tensor:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    >>> out = lucid.exp(a)
    >>> print(out)
    Tensor([[1.0, 2.7182818], [7.389056, 20.085537]], grad=None)

Performing backpropagation:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [[1.0, 2.7182818], [7.389056, 20.085537]]  # Matches the forward exponential values
