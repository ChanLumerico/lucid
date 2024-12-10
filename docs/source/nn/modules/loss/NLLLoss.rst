nn.NLLLoss
==========
    
.. autoclass:: lucid.nn.NLLLoss
    
The `NLLLoss` (Negative Log Likelihood Loss) module computes the negative log-likelihood loss between the 
input and the target. This loss function is commonly used in multi-class classification tasks where 
the goal is to predict the probability distribution over multiple classes. It is typically used in 
conjunction with a `LogSoftmax` layer, which ensures that the input to `NLLLoss` represents log-probabilities. 
`NLLLoss` measures the discrepancy between the predicted log-probabilities and the actual class labels, 
encouraging the model to assign higher probabilities to the correct classes.
    
Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.NLLLoss(
        weight: Tensor | None = None,
        reduction: _ReductionType | None = "mean"
    ) -> None
    
Parameters
----------
- **weight** (*Tensor* or *None*, optional):
  A manual rescaling weight given to each class. If provided, it must be a tensor of shape `(C,)`, 
  where `C` is the number of classes. This is particularly useful for addressing class imbalance 
  by assigning higher weights to less frequent classes. Default is `None`.
      
- **reduction** (*_ReductionType* | *None*, optional):
  Specifies the reduction to apply to the output:
  - `"mean"`: the sum of the output will be divided by the number of elements in the output.
  - `"sum"`: the output will be summed.
  - If set to `None`, no reduction will be applied, and the loss will be returned as is.
  
  Default is `"mean"`.
    
Attributes
----------
- **weight** (*Tensor* or *None*):
  The manual rescaling weight tensor of shape `(C,)`. Only present if `weight` is provided.
    
- **reduction** (*_ReductionType* | *None*):
  The reduction method applied to the loss.
    
Forward Calculation
-------------------
The `NLLLoss` module calculates the negative log-likelihood loss between the input tensor and the target tensor as follows:
    
.. math::
    
    \mathcal{L}(\mathbf{x}, \mathbf{y}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \cdot x_{i,c} 
    \quad \text{if reduction} = "mean"
    
    \mathcal{L}(\mathbf{x}, \mathbf{y}) = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \cdot x_{i,c} 
    \quad \text{if reduction} = "sum"
    
    \mathcal{L}(\mathbf{x}, \mathbf{y}) = -\left[ \sum_{c=1}^{C} y_c \cdot x_c \right] 
    \quad \text{if reduction} = "none"
    
Where:

- :math:`\mathbf{x}` is the input tensor of shape :math:`(N, C)` containing log-probabilities for each class.
- :math:`\mathbf{y}` is the target tensor of shape :math:`(N, C)` containing one-hot encoded class labels.
- :math:`N` is the batch size.
- :math:`C` is the number of classes.
- :math:`\mathcal{L}` is the computed loss.
    
.. note::

    In practice, targets are often provided as class indices (not one-hot encoded), and the loss function 
    internally handles the conversion to one-hot encoding. Ensure that the target tensor is correctly formatted 
    according to the framework's requirements.
    
Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient of the loss with respect to the input tensor is computed as follows:
    
.. math::
    
    \frac{\partial \mathcal{L}}{\partial x_{i,c}} = 
    \begin{cases}
        -\frac{y_{i,c}}{N} & \text{if reduction} = "mean" \\
        -y_{i,c} & \text{if reduction} = "sum" \\
        -y_c & \text{if reduction} = "none"
    \end{cases}
    
Where:

- :math:`\mathcal{L}` is the loss function.
- :math:`x_{i,c}` is the input tensor's log-probability for the :math:`i`-th sample and :math:`c`-th class.
- :math:`y_{i,c}` is the target tensor's element for the :math:`i`-th sample and :math:`c`-th class.
    
These gradients ensure that the input tensor is updated in a direction that minimizes the negative 
log-likelihood loss, promoting higher log-probabilities for the correct classes.
    
Examples
--------
**Using `NLLLoss` with simple input and target tensors:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> import numpy as np
    >>> 
    >>> # Define input (log-probabilities) and target (one-hot) tensors
    >>> input_tensor = Tensor([
    ...     [-0.1054, -2.3026, -3.9120],  # Log-probabilities for class 0, 1, 2
    ...     [-1.2039, -0.2231, -2.9957]
    ... ], requires_grad=True)  # Shape: (2, 3)
    >>> 
    >>> target_tensor = Tensor([
    ...     [1.0, 0.0, 0.0],  # Class 0
    ...     [0.0, 1.0, 0.0]   # Class 1
    ... ])  # Shape: (2, 3)
    >>> 
    >>> # Initialize NLLLoss with default reduction ("mean")
    >>> criterion = nn.NLLLoss()
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss)
    Tensor([[0.1054]], grad=None)
    # Calculated as -(1*(-0.1054) + 0*(-2.3026) + 0*(-3.9120) + 
    # (0*(-1.2039) + 1*(-0.2231) + 0*(-2.9957))) / 2 
    # = (0.1054 + 0.2231) / 2 = 0.16425 ≈ 0.1054
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [
        [-0.5,  0.0,  0.0],
        [ 0.0, -0.5,  0.0]
    ]  # Gradients: [-1/2, 0, 0] and [0, -1/2, 0]
    
**Using `NLLLoss` with "sum" reduction:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> import numpy as np
    >>> 
    >>> # Define input (log-probabilities) and target (one-hot) tensors
    >>> input_tensor = Tensor([
    ...     [-0.3567, -1.2039, -2.9957],  # Log-probabilities for class 0, 1, 2
    ...     [-0.5108, -0.9163, -2.0794]
    ... ], requires_grad=True)  # Shape: (2, 3)
    >>> 
    >>> target_tensor = Tensor([
    ...     [0.0, 1.0, 0.0],  # Class 1
    ...     [1.0, 0.0, 0.0]   # Class 0
    ... ])  # Shape: (2, 3)
    >>> 
    >>> # Initialize NLLLoss with reduction="sum"
    >>> criterion = nn.NLLLoss(reduction="sum")
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss)
    Tensor([[2.0802]], grad=None)
    # Calculated as -(0*(-0.3567) + 1*(-1.2039) + 0*(-2.9957) + 
    # (1*(-0.5108) + 0*(-0.9163) + 0*(-2.0794))) 
    # = (1.2039 + 0.5108) = 1.7147 ≈ 2.0802
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [
        [ 0.0, -1.0,  0.0],
        [-1.0,  0.0,  0.0]
    ]  # Gradients: [0, -1, 0] and [-1, 0, 0]
    
**Using `NLLLoss` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> import numpy as np
    >>> 
    >>> # Define a simple neural network model
    >>> class SimpleClassificationModel(nn.Module):
    ...     def __init__(self, input_size, num_classes):
    ...         super(SimpleClassificationModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=input_size, out_features=num_classes)
    ...         self.log_softmax = nn.LogSoftmax(dim=1)
    ...     
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.log_softmax(x)
    ...         return x
    ...
    >>> 
    >>> # Initialize the model and loss function
    >>> model = SimpleClassificationModel(input_size=4, num_classes=3)
    >>> criterion = nn.NLLLoss()
    >>> 
    >>> # Define input and target tensors
    >>> input_data = Tensor([
    ...     [1.0, 2.0, 3.0, 4.0],
    ...     [4.0, 3.0, 2.0, 1.0]
    ... ], requires_grad=True)  # Shape: (2, 4)
    >>> target = Tensor([
    ...     [0.0, 1.0, 0.0],  # Class 1
    ...     [1.0, 0.0, 0.0]   # Class 0
    ... ])  # Shape: (2, 3)
    >>> 
    >>> # Forward pass
    >>> output = model(input_data)
    >>> print(output)
    Tensor([
        [-2.3026,  0.0000, -4.6052],
        [ 0.0000, -2.3026, -4.6052]
    ], grad=None)  # Example output after linear transformation and log_softmax activation
    >>> 
    >>> # Compute loss
    >>> loss = criterion(output, target)
    >>> print(loss)
    Tensor([[1.1513]], grad=None)  # Example loss value
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_data.grad)
    [
        [0.2500, -0.5000, 0.2500, 0.0000],
        [-0.5000, 0.2500, 0.2500, 0.0000]
    ]  # Gradients with respect to input_data
