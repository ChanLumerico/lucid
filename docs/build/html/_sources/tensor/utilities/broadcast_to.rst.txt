lucid.broadcast_to
==================

.. autofunction:: lucid.broadcast_to

The `broadcast_to` function expands a tensor to a specified shape by replicating its dimensions 
according to broadcasting rules. This is similar to `numpy.broadcast_to`.

Function Signature
------------------

.. code-block:: python

    def broadcast_to(a: Tensor, shape: _ShapeLike) -> Tensor

Parameters
----------

- **a** (*Tensor*):
    The input tensor to be broadcasted.

- **shape** (*_ShapeLike*):
    The target shape to which the tensor should be broadcasted. 
    Must be compatible with broadcasting rules.

Returns
-------

- **Tensor**:
    A new `Tensor` with the specified shape, with data replicated as per broadcasting rules. 
    If `a` requires gradients, the returned tensor will also track gradients.

Broadcasting Rules
------------------

- The function follows NumPy broadcasting rules:

    - If the input tensor has fewer dimensions than the target shape, leading 
      dimensions are prepended with size `1`.

    - Dimensions of size `1` in the input tensor can be expanded to match the 
      corresponding dimension in the target shape.

    - Incompatible shapes will raise an error.

.. note::

   This function does not copy data; it creates a view with expanded dimensions.

Examples
--------

**Basic Example**

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([1, 2, 3])  # Shape: (3,)
    >>> b = lucid.broadcast_to(a, (2, 3))
    >>> print(b)
    Tensor([[1, 2, 3],
            [1, 2, 3]])

**Broadcasting with Extra Dimensions**

.. code-block:: python

    >>> a = lucid.Tensor([[1], [2]])  # Shape: (2, 1)
    >>> b = lucid.broadcast_to(a, (2, 3))
    >>> print(b)
    Tensor([[1, 1, 1],
            [2, 2, 2]])

Backward Gradient Calculation
-----------------------------

When performing backpropagation, gradients are summed along broadcasted dimensions:

.. math::
    \frac{\partial L}{\partial a} = \sum_{\text{broadcasted axes}} \frac{\partial L}{\partial b}

This ensures that the gradient for `a` is accumulated correctly.

**Example with Gradient Computation**

.. code-block:: python

    >>> a = lucid.Tensor([1, 2, 3], requires_grad=True)
    >>> b = lucid.broadcast_to(a, (2, 3))
    >>> b.backward(lucid.Tensor([[1, 1, 1], [1, 1, 1]]))
    >>> print(a.grad)
    [2, 2, 2]  # Sum of gradients along broadcasted dimension
