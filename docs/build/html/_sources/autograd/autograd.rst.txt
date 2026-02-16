lucid.autograd
==============

The `lucid.autograd` package provides explicit autograd utilities that mirror
common PyTorch workflows while keeping gradient calls simple and transparent.
Use this package when you want gradients returned as values or when you need
to trigger the backward engine directly without calling Tensor methods.

Overview
--------

The package offers two primary entry points:

- **grad**: Compute gradients of outputs with respect to inputs and return
  them without leaving gradients on the input tensors.
- **backward**: Run the autograd engine explicitly, mirroring the behavior of
  `Tensor.backward` while centralizing the backward logic.

Key Features
------------

.. rubric:: `Functional Gradient Extraction`

`lucid.autograd.grad` is useful for functional-style training loops, custom
loss aggregation, or when you want to avoid mutating `.grad` on inputs.

.. admonition:: Example

    .. code-block:: python

        >>> import lucid
        >>> x = lucid.tensor([1.0, -2.0, 3.0], requires_grad=True)
        >>> y = lucid.sum(x * x)
        >>> dx = lucid.autograd.grad(y, x)
        >>> dx
        array([ 2., -4.,  6.])

.. rubric:: `Explicit Engine Control`

`lucid.autograd.backward` provides direct access to the backward engine. This
is helpful for higher-level utilities that want to orchestrate backward
passes without tying logic to the Tensor class.

.. admonition:: Example

    .. code-block:: python

        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = lucid.sum(x * x)
        >>> lucid.autograd.backward(y)
        >>> x.grad
        array([2., 4., 6.])

Usage Notes
-----------

- `grad_outputs` can be used to seed gradients for vector-Jacobian products.
- By default, intermediate gradients are cleared unless `retain_grad` is set.
- The autograd utilities track first-order gradients only.

See Also
--------

- `lucid.autograd.grad`
- `lucid.autograd.backward`
