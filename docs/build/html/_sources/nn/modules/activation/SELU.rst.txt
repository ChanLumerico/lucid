nn.SELU
=======

.. autoclass:: lucid.nn.SELU
    
The `SELU` (Scaled Exponential Linear Unit) module applies the scaled exponential linear activation 
function to the input tensor. 

`SELU` is designed to induce self-normalizing properties in neural networks, maintaining the mean 
and variance of the activations close to zero and one, respectively. This helps in accelerating the 
training process and improving the performance of deep neural networks by mitigating issues like 
vanishing and exploding gradients.

Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.SELU()
    
Parameters
----------
- **None**

Attributes
----------
- **None**

Forward Calculation
-------------------
The `SELU` module performs the following operation:
    
.. math::
    
    \mathbf{y} = \lambda \times 
    \begin{cases}
        \mathbf{x} & \text{if } \mathbf{x} > 0 \\
        \alpha (\exp(\mathbf{x}) - 1) & \text{otherwise}
    \end{cases}
    
Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor after applying the SELU activation.
- :math:`\alpha` is a predefined constant, typically set to `1.67326`.
- :math:`\lambda` is a predefined scaling constant, typically set to `1.0507`.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:
    
.. math::
    
    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
    \lambda \times 
    \begin{cases}
        1 & \text{if } \mathbf{x} > 0 \\
        \alpha \exp(\mathbf{x}) & \text{otherwise}
    \end{cases}
    
This means that the gradient of the loss with respect to the input is scaled by :math:`\lambda` 
and passes through unchanged for positive input values. For negative input values, 
the gradient is scaled by :math:`\lambda \times \alpha \exp(\mathbf{x})`, allowing for small, 
non-zero gradients that help in training deeper networks.

Examples
--------
**Applying `SELU` to a single input tensor:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 2.0, -0.5, 3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> selu = nn.SELU()
    >>> output = selu(input_tensor)
    >>> print(output)
    Tensor([[-1.7635,  2.0, -0.8584,  3.0]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[0.2547, 1.0, 0.2324, 1.0]])  # Gradients with respect to input_tensor

**Using `SELU` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class SimpleSELUModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleSELUModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=3, out_features=2)
    ...         self.selu = nn.SELU()
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.selu(x)
    ...         return x
    ...
    >>> model = SimpleSELUModel()
    >>> input_data = Tensor([[0.5, -1.2, 3.3]], requires_grad=True)  # Shape: (1, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[-1.7635,  2.0]], grad=None)  # Example output after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[0.2547, 1.0, 0.2324]])  # Gradients with respect to input_data

**Integrating `SELU` into a Neural Network Model:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class SELUNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(SELUNetwork, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=4, out_features=8)
    ...         self.selu = nn.SELU()
    ...         self.fc2 = nn.Linear(in_features=8, out_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.selu(x)
    ...         x = self.fc2(x)
    ...         return x
    ...
    >>> model = SELUNetwork()
    >>> input_data = Tensor([[0.5, -1.2, 3.3, 0.7]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...], [...]], grad=None)  # Output tensor after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[...], [...]])  # Gradients with respect to input_data
