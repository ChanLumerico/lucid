nn.ReLU
=======

.. autoclass:: lucid.nn.ReLU
    
The `ReLU` (Rectified Linear Unit) module applies the rectified linear activation function to the 
input tensor. It sets all negative values in the input to zero while keeping positive values 
unchanged. This activation function introduces non-linearity into the model, enabling it to 
learn complex patterns in the data. `ReLU` is widely used in various neural network architectures 
due to its simplicity and effectiveness.

Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.ReLU()
    
Parameters
----------
- **None**

Attributes
----------
- **None**

Forward Calculation
-------------------
The `ReLU` module performs the following operation:
    
.. math::
    
    \mathbf{y} = \max(0, \mathbf{x})
    
Where:
    
- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor, where each element is the maximum of zero and the 
  corresponding input element.
    
Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:
    
.. math::
    
    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
    \begin{cases}
        1 & \text{if } \mathbf{x} > 0 \\
        0 & \text{otherwise}
    \end{cases}
    
This means that the gradient of the loss with respect to the input is passed through only for 
positive input values.

Examples
--------
**Applying `ReLU` to a single input tensor:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 2.0, -0.5, 3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> relu = nn.ReLU()
    >>> output = relu(input_tensor)
    >>> print(output)
    Tensor([[0.0, 2.0, 0.0, 3.0]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[0.0, 1.0, 0.0, 1.0]])  # Gradients with respect to input_tensor

**Using `ReLU` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class SimpleReLUModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleReLUModel, self).__init__()
    ...         self.relu = nn.ReLU()
    ...
    ...     def forward(self, x):
    ...         return self.relu(x)
    ...
    >>> model = SimpleReLUModel()
    >>> input_data = Tensor([[-2.0, 0.5, 1.5, -0.3]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[0.0, 0.5, 1.5, 0.0]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[0.0, 1.0, 1.0, 0.0]])  # Gradients with respect to input_data

**Integrating `ReLU` into a Neural Network Model:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class NeuralNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(NeuralNetwork, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=3, out_features=5)
    ...         self.relu = nn.ReLU()
    ...         self.fc2 = nn.Linear(in_features=5, out_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.relu(x)
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
    Tensor([[...]])  # Gradients with respect to input_data
