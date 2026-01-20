lucid.autograd.backward
=======================

.. autofunction:: lucid.autograd.backward

`lucid.autograd.backward` runs the internal autograd engine, mirroring the
behavior of `Tensor.backward` while centralizing the backward logic.
It is useful when you need to explicitly invoke the engine without calling
the Tensor method.

Function Signature
------------------

.. code-block:: python

    def backward(
        tensor: Tensor,
        retain_grad: bool = False,
        retain_graph: bool = False,
    ) -> None

Parameters
----------

- **tensor**: The output Tensor to backpropagate from.
- **retain_grad**: Whether to keep non-leaf gradients.
- **retain_graph**: Whether to keep the graph for further backward passes.

Behavior Notes
--------------

- Seeds the output gradient with ones if `tensor.grad` is not already set.
- Clears intermediate nodes and operation results unless `retain_graph` is set.
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
    >>> y.grad = np.array([10.0])
    >>> lucid.autograd.backward(y)
    >>> x.grad
    array([20., 40., 60.])
