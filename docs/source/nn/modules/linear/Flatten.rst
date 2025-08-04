nn.Flatten
==========

.. autoclass:: lucid.nn.Flatten

The `Flatten` module reshapes a contiguous range of dimensions in the input tensor
into a single dimension, effectively flattening part of the tensor shape.
This is commonly used to flatten spatial dimensions (like `(C, H, W)` into a single 
feature vector) before feeding into a fully connected layer.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Flatten(start_axis: int = 0, end_axis: int = -1)

Parameters
----------
- **start_axis** (*int*, optional):  
  First axis to include in the flattening. Defaults to `0`.

- **end_axis** (*int*, optional):  
  Last axis to include in the flattening. Defaults to `-1`, 
  which means the last dimension.

Forward Calculation
-------------------
The `Flatten` module transforms an input tensor of shape:

.. math::

    (d_0, d_1, \dots, d_{start-1}, d_{start}, \dots, d_{end}, d_{end+1}, \dots, d_n)

into a tensor of shape:

.. math::

    (d_0, d_1, \dots, d_{start-1}, D, d_{end+1}, \dots, d_n)

where:

.. math::

    D = \prod_{i=start}^{end} d_i

All other dimensions remain unchanged.

Backward Gradient Calculation
-----------------------------
Gradients are reshaped to the original shape during backpropagation. 
No gradients are lost or altered — the reshape operation is differentiable.

Examples
--------

**Flattening from the second dimension to the last:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)  # Shape: (2, 2, 2)
    >>> flatten = nn.Flatten(start_axis=1)
    >>> output = flatten(input_tensor)  # Shape: (2, 4)
    >>> print(output)
    Tensor([[1. 2. 3. 4.],
            [5. 6. 7. 8.]], grad=None)

**Flattening with explicit range:**

.. code-block:: python

    >>> input_tensor = Tensor([[[1, 2], [3, 4]]], requires_grad=True)  # Shape: (1, 2, 2)
    >>> flatten = nn.Flatten(start_axis=0, end_axis=1)
    >>> output = flatten(input_tensor)  # Shape: (2, 2)
    >>> print(output)
    Tensor([[1. 2.],
            [3. 4.]], grad=None)

**Using `Flatten` in a model before a linear layer:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class FlattenNet(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.flatten = nn.Flatten(start_axis=1)
    ...         self.linear = nn.Linear(in_features=784, out_features=10)
    ...
    ...     def forward(self, x):
    ...         x = self.flatten(x)
    ...         x = self.linear(x)
    ...         return x
    >>>
    >>> model = FlattenNet()
    >>> input_tensor = Tensor([[range(28)] * 28], requires_grad=True)  # Shape: (1, 28, 28)
    >>> output = model(input_tensor)
    >>> print(output)
    Tensor([[...]], grad=None)
