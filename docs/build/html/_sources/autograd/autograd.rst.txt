Autograd Engine
===============

Lucid's autograd utilities provide explicit access to gradient computation
outside of the Tensor method surface. This is useful when you want to
compute gradients without mutating input tensors, drive custom training
loops, or integrate gradient calls into higher-level utilities.

These helpers are designed to mirror common PyTorch workflows while staying
lightweight and explicit. Use `lucid.autograd.grad` when you need gradients
as return values, and `lucid.autograd.backward` when you want to run the
engine directly.

The APIs intentionally track only first-order gradients. Higher-order
gradient graphs and create-graph style features are not exposed in Lucid's
autograd helpers at this time.

Quick Start
-----------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = lucid.sum(x * x)
    >>> lucid.autograd.grad(y, x)
    array([2., 4., 6.])
