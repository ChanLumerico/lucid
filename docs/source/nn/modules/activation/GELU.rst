nn.GELU
=======

.. autoclass:: lucid.nn.GELU
    
The `GELU` (Gaussian Error Linear Unit) module applies the Gaussian Error Linear Unit 
activation function to the input tensor. `GELU` is a smooth, non-linear activation function 
that combines properties of both linear and non-linear functions. 

It is defined as:
    
.. math::
    
    \mathbf{y} = \mathbf{x} \cdot \Phi(\mathbf{x})
    
where :math:`\Phi(\mathbf{x})` is the cumulative distribution function of the standard 
normal distribution. `GELU` introduces non-linearity while maintaining differentiability, 
which can lead to improved performance in deep neural networks.

Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.GELU()
    
Parameters
----------
- **None**

Attributes
----------
- **None**

Forward Calculation
-------------------
The `GELU` module performs the following operation:
    
.. math::
    
    \mathbf{y} = \mathbf{x} \cdot \Phi(\mathbf{x})
    
Where:
- :math:`\mathbf{x}` is the input tensor.
- :math:`\Phi(\mathbf{x})` is the cumulative distribution function of the standard normal distribution.
- :math:`\mathbf{y}` is the output tensor after applying the GELU activation.
    
Alternatively, `GELU` can be approximated as:
    
.. math::
    
    \mathbf{y} = 0.5 \mathbf{x} \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} 
    \left(\mathbf{x} + 0.044715 \mathbf{x}^3\right)\right)\right)
    
Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:
    
.. math::
    
    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \Phi(\mathbf{x}) + \mathbf{x} \phi(\mathbf{x})
    
Where:
- :math:`\phi(\mathbf{x})` is the probability density function of the standard normal distribution.
    
This means that the gradient of the loss with respect to the input incorporates both the 
activation and its derivative, allowing for effective gradient flow during training.

Examples
--------
**Applying `GELU` to a single input tensor:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 0.0, 1.0, 2.0]], requires_grad=True)  # Shape: (1, 4)
    >>> gelu = nn.GELU()
    >>> output = gelu(input_tensor)
    >>> print(output)
    Tensor([[-0.1588, 0.0, 0.8413, 1.9545]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[0.1588, 0.5, 0.8413, 1.0]])  # Gradients with respect to input_tensor
    
**Using `GELU` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class SimpleGELUModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleGELUModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=3, out_features=2)
    ...         self.gelu = nn.GELU()
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.gelu(x)
    ...         return x
    ...
    >>> model = SimpleGELUModel()
    >>> input_data = Tensor([[0.5, -1.2, 3.3]], requires_grad=True)  # Shape: (1, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...], [...]], grad=None)  # Example output after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[...], [...]])  # Gradients with respect to input_data
    
**Integrating `GELU` into a Neural Network Model:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class GELUNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(GELUNetwork, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=4, out_features=8)
    ...         self.gelu = nn.GELU()
    ...         self.fc2 = nn.Linear(in_features=8, out_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.gelu(x)
    ...         x = self.fc2(x)
    ...         return x
    ...
    >>> model = GELUNetwork()
    >>> input_data = Tensor([[0.5, -1.2, 3.3, 0.7]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...], [...]], grad=None)  # Output tensor after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[...], [...]])  # Gradients with respect to input_data
