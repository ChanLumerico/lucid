lucid.autograd.backward
=======================

.. autofunction:: lucid.autograd.backward

`lucid.autograd.backward` runs the internal autograd engine, mirroring the
behavior of `Tensor.backward` while centralizing the backward logic.
It is useful for cases where you want to explicitly invoke the engine without
calling the method on a Tensor, or when building higher-level utilities that
wrap backward execution.

Unlike `lucid.autograd.grad`, this function performs a standard backward pass
and writes gradients to leaf tensors (or keeps intermediate gradients when
`retain_grad` is set).

Function Signature
------------------

.. code-block:: python

    def backward(tensor, retain_grad=False, retain_graph=False) -> None

Parameters
----------

- **tensor**: The output Tensor to backpropagate from.
- **retain_grad**: Whether to keep non-leaf gradients.
- **retain_graph**: Whether to keep the graph for further backward passes.

Behavior Notes
--------------

- Seeds the output gradient with ones if `tensor.grad` is not already set.
- Clears intermediate nodes and operation results unless `retain_graph` is
  set.
- Mirrors the same behavior as `Tensor.backward` internally.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = lucid.sum(x * x)
    >>> lucid.autograd.backward(y)
    >>> x.grad
    array([2., 4., 6.])

Manual Seed Gradient
--------------------

.. code-block:: python

    >>> import numpy as np
    >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = lucid.sum(x * x)
    >>> y.grad = np.array([10.0])  # seed gradient
    >>> lucid.autograd.backward(y)
    >>> x.grad
    array([20., 40., 60.])
