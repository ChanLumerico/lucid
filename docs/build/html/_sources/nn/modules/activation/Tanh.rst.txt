nn.Tanh
=======

.. autoclass:: lucid.nn.Tanh
    
The `Tanh` (Hyperbolic Tangent) module applies the hyperbolic tangent activation function 
to the input tensor. The `Tanh` function maps input values to the range (-1, 1), introducing 
non-linearity into the model. It is symmetric around the origin, which can help in centering 
the data and improving the convergence of the training process.

Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.Tanh()
    
Parameters
----------
- **None**

Attributes
----------
- **None**

Forward Calculation
-------------------
The `Tanh` module performs the following operation:
    
.. math::
    
    \mathbf{y} = \tanh(\mathbf{x})
    
Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor after applying the hyperbolic tangent activation function.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:
    
.. math::
    
    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 1 - \tanh^2(\mathbf{x}) = 1 - \mathbf{y}^2
    
This means that the gradient of the loss with respect to the input is scaled by the derivative of 
the `Tanh` function, allowing gradients to flow effectively during training.

Examples
--------
**Applying `Tanh` to a single input tensor:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 0.0, 1.0, 2.0]], requires_grad=True)  # Shape: (1, 4)
    >>> tanh = nn.Tanh()
    >>> output = tanh(input_tensor)
    >>> print(output)
    Tensor([[-0.7616,  0.0,  0.7616,  0.9640]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[0.4199743, 1.0, 0.4199743, 0.0706508]])  # Gradients with respect to input_tensor

**Using `Tanh` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class SimpleTanhModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleTanhModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=3, out_features=2)
    ...         self.tanh = nn.Tanh()
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.tanh(x)
    ...         return x
    ...
    >>> model = SimpleTanhModel()
    >>> input_data = Tensor([[0.5, -1.2, 3.3]], requires_grad=True)  # Shape: (1, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[0.7616, 0.9640]], grad=None)  # Example output after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[0.4199743, 0.0706508, 0.4199743]])  # Gradients with respect to input_data

**Integrating `Tanh` into a Neural Network Model:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class TanhNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(TanhNetwork, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=4, out_features=8)
    ...         self.tanh = nn.Tanh()
    ...         self.fc2 = nn.Linear(in_features=8, out_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.tanh(x)
    ...         x = self.fc2(x)
    ...         return x
    ...
    >>> model = TanhNetwork()
    >>> input_data = Tensor([[0.5, -1.2, 3.3, 0.7]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[0.7616, 0.9640]], grad=None)  # Output tensor after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[0.4199743, 0.0706508, 0.4199743, 0.0706508]])  # Gradients with respect to input_data
