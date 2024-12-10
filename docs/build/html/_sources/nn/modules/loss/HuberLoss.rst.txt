nn.HuberLoss
============

.. autoclass:: lucid.nn.HuberLoss

The `HuberLoss` module computes the Huber loss between the input and the target. 
Huber Loss is a robust loss function that is less sensitive to outliers than the 
Mean Squared Error (MSE) loss. It behaves quadratically for small errors and linearly 
for large errors, controlled by a threshold parameter :math:`\delta`. 

This makes it suitable for regression tasks where outliers may be present, 
providing a balance between sensitivity and robustness.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.HuberLoss(
        reduction: _ReductionType | None = "mean",
        delta: float = 1.0
    ) -> None

Parameters
----------
- **reduction** (*_ReductionType* | *None*, optional):
  Specifies the reduction to apply to the output:
  - `"mean"`: the sum of the output will be divided by the number of elements in the output.
  - `"sum"`: the output will be summed.
  - If set to `None`, no reduction will be applied, and the loss will be returned as is.
  
  Default is `"mean"`.

- **delta** (*float*, optional):
  The threshold at which the Huber loss function changes from quadratic to linear.
  Default is `1.0`.

Attributes
----------
- **reduction** (*_ReductionType* | *None*):
  The reduction method applied to the loss.

- **delta** (*float*):
  The threshold parameter controlling the transition from quadratic to linear loss.

Forward Calculation
-------------------
The `HuberLoss` module calculates the Huber loss between the input tensor and the 
target tensor as follows:

.. math::

    \mathcal{L}(\mathbf{x}, \mathbf{y}) = 
    \begin{cases}
        \frac{1}{2} (\mathbf{x} - \mathbf{y})^2 & \text{for } |\mathbf{x} - \mathbf{y}| \leq \delta \\
        \delta \cdot \left( |\mathbf{x} - \mathbf{y}| - \frac{1}{2} \delta \right) & \text{otherwise}
    \end{cases}
    \quad \text{if reduction} = "mean"

    \mathcal{L}(\mathbf{x}, \mathbf{y}) = 
    \begin{cases}
        \frac{1}{2} (\mathbf{x} - \mathbf{y})^2 & \text{for } |\mathbf{x} - \mathbf{y}| \leq \delta \\
        \delta \cdot \left( |\mathbf{x} - \mathbf{y}| - \frac{1}{2} \delta \right) & \text{otherwise}
    \end{cases}
    \quad \text{if reduction} = "sum"

    \mathcal{L}(\mathbf{x}, \mathbf{y}) = 
    \begin{cases}
        \frac{1}{2} (\mathbf{x} - \mathbf{y})^2 & \text{for } |\mathbf{x} - \mathbf{y}| \leq \delta \\
        \delta \cdot \left( |\mathbf{x} - \mathbf{y}| - \frac{1}{2} \delta \right) & \text{otherwise}
    \end{cases}
    \quad \text{if reduction} = "none"

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the target tensor.
- :math:`\delta` is the threshold parameter.
- :math:`\mathcal{L}` is the computed loss.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient of the loss with respect to the input tensor 
is computed as follows:

.. math::

    \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = 
    \begin{cases}
        \mathbf{x} - \mathbf{y} & \text{for } |\mathbf{x} - \mathbf{y}| \leq \delta \\
        \delta \cdot \text{sign}(\mathbf{x} - \mathbf{y}) & \text{otherwise}
    \end{cases}
    \quad \text{if reduction} = "mean"

    \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = 
    \begin{cases}
        \mathbf{x} - \mathbf{y} & \text{for } |\mathbf{x} - \mathbf{y}| \leq \delta \\
        \delta \cdot \text{sign}(\mathbf{x} - \mathbf{y}) & \text{otherwise}
    \end{cases}
    \quad \text{if reduction} = "sum"

    \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = 
    \begin{cases}
        \mathbf{x} - \mathbf{y} & \text{for } |\mathbf{x} - \mathbf{y}| \leq \delta \\
        \delta \cdot \text{sign}(\mathbf{x} - \mathbf{y}) & \text{otherwise}
    \end{cases}
    \quad \text{if reduction} = "none"

Where:

- :math:`\mathcal{L}` is the loss function.
- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the target tensor.
- :math:`\delta` is the threshold parameter.
- :math:`\text{sign}(\cdot)` is the sign function.

These gradients ensure that the input tensor is updated in a direction that 
minimizes the Huber loss, balancing sensitivity to small errors and robustness to large errors.

Examples
--------
**Using `HuberLoss` with simple input and target tensors:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define input and target tensors
    >>> input_tensor = Tensor([[1.5, 2.0, 3.5, 4.0]], requires_grad=True)  # Shape: (1, 4)
    >>> target_tensor = Tensor([[1.0, 2.5, 3.0, 5.0]])  # Shape: (1, 4)
    >>> 
    >>> # Initialize HuberLoss with default reduction ("mean")
    >>> criterion = nn.HuberLoss(delta=1.0)
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss)
    Tensor([[0.4375]], grad=None)  # Calculated based on Huber loss formula
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [[0.1250, -0.2500, 0.2500, -0.5000]]  # Gradients based on Huber loss
     
**Using `HuberLoss` with "sum" reduction:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define input and target tensors
    >>> input_tensor = Tensor([[2.0, 1.0, 3.0, 4.0]], requires_grad=True)  # Shape: (1, 4)
    >>> target_tensor = Tensor([[1.0, 2.0, 2.0, 5.0]])  # Shape: (1, 4)
    >>> 
    >>> # Initialize HuberLoss with reduction="sum"
    >>> criterion = nn.HuberLoss(reduction="sum", delta=1.0)
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss)
    Tensor([[1.3750]], grad=None)  # Calculated as sum of individual Huber losses
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [[1.0000, -1.0000, 1.0000, -1.0000]]  # Gradients based on Huber loss with sum reduction

**Using `HuberLoss` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define a simple neural network model
    >>> class HuberRegressionModel(nn.Module):
    ...     def __init__(self):
    ...         super(HuberRegressionModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=2, out_features=1)
    ...     
    ...     def forward(self, x):
    ...         return self.linear(x)
    ...
    >>> 
    >>> # Initialize the model and loss function
    >>> model = HuberRegressionModel()
    >>> criterion = nn.HuberLoss(delta=1.0)
    >>> 
    >>> # Define input and target tensors
    >>> input_data = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape: (2, 2)
    >>> target = Tensor([[2.0], [5.0]])  # Shape: (2, 1)
    >>> 
    >>> # Forward pass
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...], [...]], grad=None)  # Example output after linear transformation
    >>> 
    >>> # Compute loss
    >>> loss = criterion(output, target)
    >>> print(loss)
    Tensor([[...]], grad=None)  # Example loss value
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_data.grad)
    # Gradients with respect to input_data
