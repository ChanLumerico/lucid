nn.Swish
========

.. autoclass:: lucid.nn.Swish

The `Swish` module applies the Swish activation function to the input tensor. 
The Swish function is defined as:

.. math::

    \text{Swish}(\mathbf{x}) = \mathbf{x} \cdot \sigma(\mathbf{x})

Where :math:`\sigma(\mathbf{x})` is the sigmoid function. Swish is a smooth, 
non-monotonic activation function that has been shown to perform better than 
ReLU in certain neural network architectures.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Swish()

Forward Calculation
-------------------
The `Swish` module performs the following operation:

.. math::

    \mathbf{y} = \mathbf{x} \cdot \frac{1}{1 + e^{-\mathbf{x}}}

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor, calculated as the element-wise 
  product of the input and its sigmoid.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \sigma(\mathbf{x}) 
    \cdot (1 + \mathbf{x} \cdot (1 - \sigma(\mathbf{x})))

This formula accounts for both the input tensor and its sigmoid in the gradient computation.

Examples
--------
**Applying `Swish` to a single input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 2.0, -0.5, 3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> swish = nn.Swish()
    >>> output = swish(input_tensor)
    >>> print(output)
    Tensor([[-0.26894142, 1.76159416, -0.18877099, 2.85772238]], grad=None)

    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[...]]  # Gradients with respect to input_tensor

**Using `Swish` within a simple neural network:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class SimpleSwishModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleSwishModel, self).__init__()
    ...         self.swish = nn.Swish()
    ...
    ...     def forward(self, x):
    ...         return self.swish(x)
    ...
    >>> model = SimpleSwishModel()
    >>> input_data = Tensor([[-2.0, 0.5, 1.5, -0.3]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[-0.23840584, 0.37754067, 1.30823025, -0.07722396]], grad=None)

    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    [[...]]  # Gradients with respect to input_data

**Integrating `Swish` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class NeuralNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(NeuralNetwork, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=3, out_features=5)
    ...         self.swish = nn.Swish()
    ...         self.fc2 = nn.Linear(in_features=5, out_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.swish(x)
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
