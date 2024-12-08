nn.Sigmoid
==========

.. autoclass:: lucid.nn.Sigmoid

The `Sigmoid` module applies the sigmoid activation function to the input tensor. 
The sigmoid function maps any real-valued number into the range (0, 1), making it 
especially useful for binary classification tasks and as an activation function 
in neural networks to introduce non-linearity.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Sigmoid()

Parameters
----------
- **None**

Attributes
----------
- **None**

Forward Calculation
-------------------
The `Sigmoid` module performs the following operation:

.. math::

    \mathbf{y} = \sigma(\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{x}}}

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor after applying the sigmoid activation function.
- :math:`\sigma` denotes the sigmoid function.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{y} \cdot (1 - \mathbf{y})

This means that the gradient of the loss with respect to the input is scaled by the product 
of the sigmoid output and one minus the sigmoid output, allowing gradients to flow effectively 
during training.

Examples
--------
**Applying `Sigmoid` to a single input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 0.0, 1.0, 2.0]], requires_grad=True)  # Shape: (1, 4)
    >>> sigmoid = nn.Sigmoid()
    >>> output = sigmoid(input_tensor)
    >>> print(output)
    Tensor([[0.2689, 0.5, 0.7311, 0.8808]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[0.1966, 0.25, 0.1966, 0.1050]])  # Gradients with respect to input_tensor

**Using `Sigmoid` within a simple neural network:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class SimpleSigmoidModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleSigmoidModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=3, out_features=2)
    ...         self.sigmoid = nn.Sigmoid()
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.sigmoid(x)
    ...         return x
    ...
    >>> model = SimpleSigmoidModel()
    >>> input_data = Tensor([[0.5, -1.2, 3.3]], requires_grad=True)  # Shape: (1, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[0.7311, 0.8808]], grad=None)  # Example output after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[0.1966, 0.1050, 0.1966]])  # Gradients with respect to input_data

**Integrating `Sigmoid` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class SigmoidNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(SigmoidNetwork, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=4, out_features=8)
    ...         self.sigmoid = nn.Sigmoid()
    ...         self.fc2 = nn.Linear(in_features=8, out_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.sigmoid(x)
    ...         x = self.fc2(x)
    ...         return x
    ...
    >>> model = SigmoidNetwork()
    >>> input_data = Tensor([[0.5, -1.2, 3.3, 0.7]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[0.7311, 0.8808]], grad=None)  # Output tensor after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[0.1966, 0.1050, 0.1966, 0.1050]])  # Gradients with respect to input_data
