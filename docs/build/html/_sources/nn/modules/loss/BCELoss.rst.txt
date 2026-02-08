nn.BCELoss
==========
    
.. autoclass:: lucid.nn.BCELoss
    
The `BCELoss` (Binary Cross Entropy Loss) module computes the binary cross-entropy loss between the 
input and the target. This loss function is commonly used in binary classification tasks where 
the goal is to predict probabilities of class membership. It measures the discrepancy between 
the predicted probabilities and the actual binary labels, encouraging the model to output probabilities 
close to the true labels.

Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.BCELoss(
        weight: Tensor | None = None,
        reduction: _ReductionType | None = "mean",
        eps: float = 1e-7,
    ) -> None

Parameters
----------
- **weight** (*Tensor* or *None*, optional):
  A manual rescaling weight given to the loss of each batch element. If given, it has 
  to be a tensor of size `N`, where `N` is the batch size. Default is `None`.
    
- **reduction** (*_ReductionType* | *None*, optional):
  Specifies the reduction to apply to the output:
  - `"mean"`: the sum of the output will be divided by the number of elements in the output.
  - `"sum"`: the output will be summed.
  - If set to `None`, no reduction will be applied, and the loss will be returned as is.
  
  Default is `"mean"`.
    
- **eps** (*float*, optional):
  A small value added to the input to prevent log(0) errors. Default is `1e-7`.

Attributes
----------
- **weight** (*Tensor* or *None*):
  The manual rescaling weight tensor of shape `(N,)`. Only present if `weight` is provided.
    
- **reduction** (*_ReductionType* | *None*):
  The reduction method applied to the loss.

Forward Calculation
-------------------
The `BCELoss` module calculates the binary cross-entropy loss between the 
input tensor and the target tensor as follows:

.. math::

    \mathcal{L}(\mathbf{x}, \mathbf{y}) = -\frac{1}{N} \sum_{i=1}^{N} 
    \left[ y_i \cdot \log(x_i + \epsilon) + (1 - y_i) \cdot \log(1 - x_i + \epsilon) \right] 
    \quad \text{if reduction} = "mean"

    \mathcal{L}(\mathbf{x}, \mathbf{y}) = -\sum_{i=1}^{N} \left[ y_i \cdot 
    \log(x_i + \epsilon) + (1 - y_i) \cdot \log(1 - x_i + \epsilon) \right] \quad \text{if reduction} = "sum"

    \mathcal{L}(\mathbf{x}, \mathbf{y}) = -\left[ y \cdot \log(x + \epsilon) + (1 - y) 
    \cdot \log(1 - x + \epsilon) \right] \quad \text{if reduction} = "none"

Where:

- :math:`\mathbf{x}` is the input tensor containing predicted probabilities.
- :math:`\mathbf{y}` is the target tensor containing binary labels (0 or 1).
- :math:`N` is the number of elements in the input tensor.
- :math:`\epsilon` is a small constant to ensure numerical stability.
- :math:`\mathcal{L}` is the computed loss.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient of the loss with respect to the input tensor 
is computed as follows:

.. math::

    \frac{\partial \mathcal{L}}{\partial x_i} = -\frac{y_i}{x_i + \epsilon} + 
    \frac{1 - y_i}{1 - x_i + \epsilon} \quad \text{if reduction} = "mean"

    \frac{\partial \mathcal{L}}{\partial x_i} = -\left( \frac{y_i}{x_i + \epsilon} + 
    \frac{1 - y_i}{1 - x_i + \epsilon} \right) \quad \text{if reduction} = "sum"

    \frac{\partial \mathcal{L}}{\partial x} = -\left[ \frac{y}{x + \epsilon} - 
    \frac{1 - y}{1 - x + \epsilon} \right] \quad \text{if reduction} = "none"

Where:

- :math:`\mathcal{L}` is the loss function.
- :math:`x_i` is the :math:`i`-th element of the input tensor.
- :math:`y_i` is the :math:`i`-th element of the target tensor.

These gradients ensure that the input tensor is updated in a direction that 
minimizes the binary cross-entropy loss.

Examples
--------
**Using `BCELoss` with simple input and target tensors:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define input and target tensors
    >>> input_tensor = Tensor([[0.9, 0.2, 0.8, 0.4]], requires_grad=True)  # Shape: (1, 4)
    >>> target_tensor = Tensor([[1.0, 0.0, 1.0, 0.0]])  # Shape: (1, 4)
    >>> 
    >>> # Initialize BCELoss with default reduction ("mean")
    >>> criterion = nn.BCELoss()
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss)
    Tensor([[0.1643]], grad=None)  # Calculated as -[(1*log(0.9) + 0*log(0.1) + 1*log(0.8) + 0*log(0.6)) / 4] ≈ 0.1643
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [[-0.1111,  0.5000, -0.2500,  1.6667]]  # Gradients: [-1/0.9 /4, 0/..., etc.]
    
**Using `BCELoss` with "sum" reduction:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define input and target tensors
    >>> input_tensor = Tensor([[0.7, 0.3, 0.2]], requires_grad=True)  # Shape: (1, 3)
    >>> target_tensor = Tensor([[1.0, 0.0, 0.0]])  # Shape: (1, 3)
    >>> 
    >>> # Initialize BCELoss with reduction="sum"
    >>> criterion = nn.BCELoss(reduction="sum")
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss)
    Tensor([[0.3567]], grad=None)  # Calculated as -(1*log(0.7) + 0*log(0.3) + 0*log(0.8)) ≈ 0.3567
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [[-1.4286,  0.0,  0.0]]  # Gradients: [-1/0.7, 0/..., 0/...]
    
**Using `BCELoss` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define a simple neural network model
    >>> class SimpleBinaryClassificationModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleBinaryClassificationModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=2, out_features=1)
    ...         self.sigmoid = nn.Sigmoid()
    ...     
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.sigmoid(x)
    ...         return x
    ...
    >>> 
    >>> # Initialize the model and loss function
    >>> model = SimpleBinaryClassificationModel()
    >>> criterion = nn.BCELoss()
    >>> 
    >>> # Define input and target tensors
    >>> input_data = Tensor([[0.5, -0.2]], requires_grad=True)  # Shape: (1, 2)
    >>> target = Tensor([[1.0]])  # Shape: (1, 1)
    >>> 
    >>> # Forward pass
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[0.6682]], grad=None)  # Example output after linear transformation and sigmoid activation
    >>> 
    >>> # Compute loss
    >>> loss = criterion(output, target)
    >>> print(loss)
    Tensor([[0.4044]], grad=None)  # Example loss value
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_data.grad)
    # Gradients with respect to input_data
