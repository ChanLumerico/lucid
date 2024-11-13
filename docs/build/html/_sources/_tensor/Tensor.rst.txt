Tensor
======

Class Definition
----------------

.. autoclass:: lucid._tensor.Tensor
    :members:
    :undoc-members:
    :no-inherited-members:

Usage
-----

This `Tensor` class is initialized with data, and optionally allows setting flags for gradient requirements and data type. It includes gradient tracking and a `backward` method to compute gradients for operations that use the tensor.

Examples
--------

Creating a tensor, setting gradients, and computing a backward pass:

.. code-block:: python

    import numpy as np
    from lucid import Tensor

    # Initialize tensor with data and gradient tracking
    a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
    
    # Perform an operation
    c = a + b

    # Compute gradients
    c.backward()

    # Access gradients
    print(a.grad)  # Displays gradient with respect to `a`
    print(b.grad)  # Displays gradient with respect to `b`

.. note::

    If `requires_grad` is set to `False`, the tensor will not track gradients or participate in the computational graph. Leaf tensors in the graph are nodes that are not derived from other tensors, and gradients can only be directly assigned to these leaf nodes.
