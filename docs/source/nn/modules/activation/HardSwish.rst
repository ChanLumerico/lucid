nn.HardSwish
============

.. autoclass:: lucid.nn.HardSwish

The `HardSwish` module applies the HardSwish activation function to the input tensor. 
The HardSwish function is defined as:

.. math::

    \text{HardSwish}(\mathbf{x}) = \mathbf{x} \cdot \text{HardSigmoid}(\mathbf{x})

Where :math:`\text{HardSigmoid}(\mathbf{x})` is a piecewise linear approximation of 
the sigmoid function:

.. math::

    \text{HardSigmoid}(\mathbf{x}) = \max(0, \min(1, 0.167 \cdot \mathbf{x} + 0.5))

The HardSwish activation function is computationally efficient and is commonly used in 
lightweight neural network architectures.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.HardSwish()

Forward Calculation
-------------------
The `HardSwish` module performs the following operation:

.. math::

    \mathbf{y} = \mathbf{x} \cdot \max(0, \min(1, 0.167 \cdot \mathbf{x} + 0.5))

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor, calculated as the element-wise 
- product of the input and the hard sigmoid.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
    \begin{cases}
        0.167 \cdot \mathbf{x} + 0.5 + 0.167 \cdot \mathbf{x} & \text{if } 0 \leq 0.167 \cdot \mathbf{x} + 0.5 \leq 1 \\
        1 & \text{if } 0.167 \cdot \mathbf{x} + 0.5 > 1 \\
        0 & \text{otherwise}
    \end{cases}

Examples
--------
**Applying `HardSwish` to a single input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 2.0, -0.5, 3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> hardswish = nn.HardSwish()
    >>> output = hardswish(input_tensor)
    >>> print(output)
    Tensor([[0.0, 2.0, 0.0, 3.0]], grad=None)

    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[...]]  # Gradients with respect to input_tensor

**Using `HardSwish` within a simple neural network:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class SimpleHardSwishModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleHardSwishModel, self).__init__()
    ...         self.hardswish = nn.HardSwish()
    ...
    ...     def forward(self, x):
    ...         return self.hardswish(x)
    ...
    >>> model = SimpleHardSwishModel()
    >>> input_data = Tensor([[-2.0, 0.5, 1.5, -0.3]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[0.0, 0.3, 1.2, 0.0]], grad=None)

    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    [[...]]  # Gradients with respect to input_data

**Integrating `HardSwish` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class NeuralNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(NeuralNetwork, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=3, out_features=5)
    ...         self.hardswish = nn.HardSwish()
    ...         self.fc2 = nn.Linear(in_features=5, out_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.hardswish(x)
    ...         x = self.fc2(x)
    ...         return x
    ...
    >>> model = NeuralNetwork()
    >>> input_data = Tensor([[0.5, -1.2, 3.3]], requires_grad=True)  # Shape: (1, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...]], grad=None)  # Output tensor after passing through the model

    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    [[...]]  # Gradients with respect to input_data
