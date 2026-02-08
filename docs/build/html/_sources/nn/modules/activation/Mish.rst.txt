nn.Mish
=======

.. autoclass:: lucid.nn.Mish

The `Mish` module applies the Mish activation function to the input tensor. 
The Mish function is defined as:

.. math::

    \text{Mish}(\mathbf{x}) = \mathbf{x} \cdot \tanh(\ln(1 + e^{\mathbf{x}}))

Where :math:`\tanh(\cdot)` is the hyperbolic tangent function and 
:math:`\ln(1 + e^{\mathbf{x}})` is the softplus function. Mish is a smooth, 
non-monotonic activation function that has demonstrated promising performance 
in deep neural networks.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Mish()

Forward Calculation
-------------------
The `Mish` module performs the following operation:

.. math::

    \mathbf{y} = \mathbf{x} \cdot \tanh(\ln(1 + e^{\mathbf{x}}))

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor, calculated as the element-wise product 
  of the input and the hyperbolic tangent of its softplus.

Backward Gradient Calculation
-----------------------------
During backpropagation, the derivative of Mish with respect to the input is:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \tanh(\text{sp}) + 
    \mathbf{x} \cdot \sigma(\mathbf{x}) \cdot (1 - \tanh^2(\text{sp}))

Where:

- :math:`\text{sp} = \ln(1 + e^{\mathbf{x}})` is the softplus of the input.
- :math:`\sigma(\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{x}}}` is the sigmoid function.
- This derivative accounts for both the input and its transformation under softplus and tanh.

The Lucid autodiff engine uses this formula internally for accurate and efficient 
gradient computation.

Examples
--------

**Applying `Mish` to a single input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 2.0, -0.5, 3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> mish = nn.Mish()
    >>> output = mish(input_tensor)
    >>> print(output)
    Tensor([[-0.3034016, 1.9439592, -0.2291952, 2.9865277]], grad=None)

    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[...]]  # Gradients with respect to input_tensor

**Using `Mish` within a simple neural network:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class SimpleMishModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleMishModel, self).__init__()
    ...         self.mish = nn.Mish()
    ...
    ...     def forward(self, x):
    ...         return self.mish(x)
    ...
    >>> model = SimpleMishModel()
    >>> input_data = Tensor([[-2.0, 0.5, 1.5, -0.3]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[-0.2525015, 0.4078608, 1.416362, -0.086727]], grad=None)

    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    [[...]]  # Gradients with respect to input_data

**Integrating `Mish` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class NeuralNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(NeuralNetwork, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=3, out_features=5)
    ...         self.mish = nn.Mish()
    ...         self.fc2 = nn.Linear(in_features=5, out_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.mish(x)
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
