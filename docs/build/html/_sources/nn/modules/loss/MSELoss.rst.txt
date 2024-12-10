nn.MSELoss
==========
    
.. autoclass:: lucid.nn.MSELoss
    
The `MSELoss` module computes the Mean Squared Error (MSE) loss between the input and the target. 
MSELoss measures the average of the squares of the errors, that is, the average squared difference 
between the estimated values and the actual value. It is commonly used in regression tasks 
where the objective is to minimize the difference between the predicted and true values.
    
Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.MSELoss(
        reduction: _ReductionType | None = "mean"
    ) -> None
    
Parameters
----------
- **reduction** (*_ReductionType* | *None*, optional):
  Specifies the reduction to apply to the output:
  - `"mean"`: the sum of the squared errors is divided by the number of elements.
  - `"sum"`: the squared errors are summed.
  - If set to `None`, no reduction is applied, and the loss is returned as is.
  
  Default is `"mean"`.
    
Attributes
----------
- **reduction** (*_ReductionType* | *None*):
  The reduction method applied to the loss.

Forward Calculation
-------------------
The `MSELoss` module calculates the loss between the input tensor and 
the target tensor as follows:
    
.. math::
    
    \mathcal{L}(\mathbf{x}, \mathbf{y}) = \frac{1}{N} \sum_{i=1}^{N} (\mathbf{x}_i - 
    \mathbf{y}_i)^2 \quad \text{if reduction} = "mean"
    
    \mathcal{L}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{N} (\mathbf{x}_i - \mathbf{y}_i)^2 
    \quad \text{if reduction} = "sum"
    
Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the target tensor.
- :math:`N` is the number of elements in the input tensor.
- :math:`\mathcal{L}` is the computed loss.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient of the loss with respect to the input tensor is computed as follows:
    
.. math::
    
    \frac{\partial \mathcal{L}}{\partial \mathbf{x}_i} = 
    \begin{cases}
        \frac{2}{N} (\mathbf{x}_i - \mathbf{y}_i) & \text{if reduction} = "mean" \\
        2 (\mathbf{x}_i - \mathbf{y}_i) & \text{if reduction} = "sum"
    \end{cases}
    
Where:

- :math:`\mathbf{x}_i` is the :math:`i`-th element of the input tensor.
- :math:`\mathbf{y}_i` is the :math:`i`-th element of the target tensor.
- :math:`\mathcal{L}` is the loss function.
    
These gradients ensure that the input tensor is updated in the direction that minimizes the MSE loss.

Examples
--------
**Using `MSELoss` with simple input and target tensors:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define input and target tensors
    >>> input_tensor = Tensor([[2.5, 0.0, 2.0, 8.0]], requires_grad=True)  # Shape: (1, 4)
    >>> target_tensor = Tensor([[3.0, -0.5, 2.0, 7.0]])  # Shape: (1, 4)
    >>> 
    >>> # Initialize MSELoss with default reduction ("mean")
    >>> criterion = nn.MSELoss()
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss.item())
    0.375  # Calculated as ((2.5-3)^2 + (0 - (-0.5))^2 + (2-2)^2 + (8-7)^2) / 4 = (0.25 + 0.25 + 0 + 1) / 4 = 1.5 / 4 = 0.375
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [[-0.5,  0.5,  0.0,  0.5]]  # Gradients: [2*(2.5-3)/4, 2*(0 - (-0.5))/4, 2*(2-2)/4, 2*(8-7)/4] = [-0.5, 0.5, 0.0, 0.5]
    
**Using `MSELoss` with "sum" reduction:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define input and target tensors
    >>> input_tensor = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # Shape: (1, 3)
    >>> target_tensor = Tensor([[1.5, 2.5, 3.5]])  # Shape: (1, 3)
    >>> 
    >>> # Initialize MSELoss with reduction="sum"
    >>> criterion = nn.MSELoss(reduction="sum")
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss.item())
    0.75  # Calculated as (1.0-1.5)^2 + (2.0-2.5)^2 + (3.0-3.5)^2 = 0.25 + 0.25 + 0.25 = 0.75
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [[-1.0, -1.0, -1.0]]  # Gradients: [2*(1.0-1.5), 2*(2.0-2.5), 2*(3.0-3.5)] = [-1.0, -1.0, -1.0]
    
**Using `MSELoss` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define a simple neural network model
    >>> class SimpleRegressionModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleRegressionModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=2, out_features=1)
    ...     
    ...     def forward(self, x):
    ...         return self.linear(x)
    ...
    >>> 
    >>> # Initialize the model and loss function
    >>> model = SimpleRegressionModel()
    >>> criterion = nn.MSELoss()
    >>> 
    >>> # Define input and target tensors
    >>> input_data = Tensor([[1.0, 2.0]], requires_grad=True)  # Shape: (1, 2)
    >>> target = Tensor([[3.0]])  # Shape: (1, 1)
    >>> 
    >>> # Forward pass
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...]], grad=None)  # Example output after linear transformation
    >>> 
    >>> # Compute loss
    >>> loss = criterion(output, target)
    >>> print(loss.item())
    # Example loss value
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_data.grad)
    # Gradients with respect to input_data
