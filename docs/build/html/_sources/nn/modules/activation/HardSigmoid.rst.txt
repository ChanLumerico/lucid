nn.HardSigmoid
==============

.. autoclass:: lucid.nn.HardSigmoid
    
The `HardSigmoid` module applies a piecewise linear approximation of the sigmoid activation function. 
It is computationally efficient and commonly used in neural network architectures where simplicity 
and speed are prioritized.

Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.HardSigmoid()

Forward Calculation
-------------------
The `HardSigmoid` module performs the following operation:
    
.. math::
    
    \mathbf{y} = \max(0, \min(1, 0.2 \cdot \mathbf{x} + 0.5))
    
Where:
    
- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor, where each element is the result of scaling, shifting, 
  and clipping the corresponding input element.
    
Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:
    
.. math::
    
    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
    \begin{cases}
        0.2 & \text{if } 0 \leq 0.2 \cdot \mathbf{x} + 0.5 \leq 1 \\
        0 & \text{otherwise}
    \end{cases}
    
This means that the gradient is non-zero only within the linear 
region of the `HardSigmoid` operation.

Examples
--------
**Applying `HardSigmoid` to a single input tensor:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 2.0, -0.5, 3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> hard_sigmoid = nn.HardSigmoid()
    >>> output = hard_sigmoid(input_tensor)
    >>> print(output)
    Tensor([[0.3, 1.0, 0.4, 1.0]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[0.2, 0.2, 0.2, 0.0]]  # Gradients with respect to input_tensor

**Using `HardSigmoid` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class SimpleHardSigmoidModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleHardSigmoidModel, self).__init__()
    ...         self.hard_sigmoid = nn.HardSigmoid()
    ...
    ...     def forward(self, x):
    ...         return self.hard_sigmoid(x)
    ...
    >>> model = SimpleHardSigmoidModel()
    >>> input_data = Tensor([[-2.0, 0.5, 1.5, -0.3]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[0.0, 0.6, 0.8, 0.4]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    [[0.0, 0.2, 0.2, 0.2]]  # Gradients with respect to input_data

**Integrating `HardSigmoid` into a Neural Network Model:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class NeuralNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(NeuralNetwork, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=3, out_features=5)
    ...         self.hard_sigmoid = nn.HardSigmoid()
    ...         self.fc2 = nn.Linear(in_features=5, out_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.hard_sigmoid(x)
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
    [[...]] # Gradients with respect to input_data
