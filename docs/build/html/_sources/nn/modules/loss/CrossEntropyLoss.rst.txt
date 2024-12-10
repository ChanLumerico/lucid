nn.CrossEntropyLoss
===================
    
.. autoclass:: lucid.nn.CrossEntropyLoss
    
The `CrossEntropyLoss` module computes the cross-entropy loss between the input and the target. 
This loss function is widely used in multi-class classification tasks where the goal is to 
predict the probability distribution over multiple classes. 

It combines `LogSoftmax` and `Negative Log Likelihood Loss` in a single class, making it 
numerically stable and efficient. `CrossEntropyLoss` measures the discrepancy between the 
predicted probability distribution and the actual distribution (usually represented by class 
indices), encouraging the model to assign higher probabilities to the correct classes.
    
Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.CrossEntropyLoss(
        weight: Tensor | None = None,
        reduction: _ReductionType | None = "mean",
        eps: float = 1e-7,
    ) -> None
    
Parameters
----------
- **weight** (*Tensor* or *None*, optional):
  A manual rescaling weight given to each class. If given, it has to be a tensor of size 
  `C`, where `C` is the number of classes. This is particularly useful for addressing 
  class imbalance by assigning higher weights to less frequent classes. Default is `None`.
    
- **reduction** (*_ReductionType* | *None*, optional):
  Specifies the reduction to apply to the output:
  - `"mean"`: the sum of the output will be divided by the number of elements in the output.
  - `"sum"`: the output will be summed.
  - If set to `None`, no reduction will be applied, and the loss will be returned as is.

  Default is `"mean"`.
    
- **eps** (*float*, optional):
  A small value added to the input to prevent log(0) errors, ensuring numerical stability.
  Default is `1e-7`.
    
Attributes
----------
- **weight** (*Tensor* or *None*):
  The manual rescaling weight tensor of shape `(C,)`. Only present if `weight` is provided.
    
- **reduction** (*_ReductionType* | *None*):
  The reduction method applied to the loss.
    
Forward Calculation
-------------------
The `CrossEntropyLoss` module calculates the cross-entropy loss between the input tensor 
and the target tensor as follows:
    
.. math::
    
    \mathcal{L}(\mathbf{x}, \mathbf{y}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \cdot 
    \log\left(\frac{e^{x_{i,c}}}{\sum_{k=1}^{C} e^{x_{i,k}}} + \epsilon\right) 
    \quad \text{if reduction} = "mean"
    
    \mathcal{L}(\mathbf{x}, \mathbf{y}) = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \cdot 
    \log\left(\frac{e^{x_{i,c}}}{\sum_{k=1}^{C} e^{x_{i,k}}} + \epsilon\right) 
    \quad \text{if reduction} = "sum"
    
    \mathcal{L}(\mathbf{x}, \mathbf{y}) = -\left[ \sum_{c=1}^{C} y_c \cdot 
    \log\left(\frac{e^{x_c}}{\sum_{k=1}^{C} e^{x_k}} + \epsilon\right) \right] 
    \quad \text{if reduction} = "none"
    
Where:

- :math:`\mathbf{x}` is the input tensor of shape :math:`(N, C)` containing raw scores (logits) for each class.
- :math:`\mathbf{y}` is the target tensor of shape :math:`(N, C)` containing one-hot encoded class labels.
- :math:`N` is the batch size.
- :math:`C` is the number of classes.
- :math:`\epsilon` is a small constant to ensure numerical stability.
- :math:`\mathcal{L}` is the computed loss.

.. note::

    In practice, targets are often provided as class indices, and the loss function internally 
    handles the conversion to one-hot encoding. Ensure that the target tensor is correctly formatted 
    according to the framework's requirements.
    
Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient of the loss with respect to the input tensor 
is computed as follows:
    
.. math::
    
    \frac{\partial \mathcal{L}}{\partial x_{i,c}} = 
    \frac{e^{x_{i,c}}}{\sum_{k=1}^{C} e^{x_{i,k}} + \epsilon} \cdot \left( y_{i,c} - 
    \frac{e^{x_{i,c}}}{\sum_{k=1}^{C} e^{x_{i,k}} + \epsilon} \right) 
    \quad \text{if reduction} = "mean" \text{ or } "sum"
    
    \frac{\partial \mathcal{L}}{\partial x_c} = 
    \frac{e^{x_c}}{\sum_{k=1}^{C} e^{x_k} + \epsilon} \cdot \left( y_c - 
    \frac{e^{x_c}}{\sum_{k=1}^{C} e^{x_k} + \epsilon} \right) 
    \quad \text{if reduction} = "none"
    
Where:

- :math:`\mathcal{L}` is the loss function.
- :math:`x_{i,c}` is the input tensor's element for the :math:`i`-th sample and :math:`c`-th class.
- :math:`y_{i,c}` is the target tensor's element for the :math:`i`-th sample and :math:`c`-th class.
- The gradients are scaled appropriately based on the reduction method.
    
These gradients ensure that the input tensor is updated in a direction that minimizes the cross-entropy loss, 
promoting higher probabilities for the correct classes.
    
Examples
--------
**Using `CrossEntropyLoss` with simple input and target tensors:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> import numpy as np
    >>> 
    >>> # Define input (logits) and target (one-hot) tensors
    >>> input_tensor = Tensor([
    ...     [2.0, 1.0, 0.1],
    ...     [0.5, 2.5, 0.3]
    ... ], requires_grad=True)  # Shape: (2, 3)
    >>> 
    >>> target_tensor = Tensor([
    ...     [1.0, 0.0, 0.0],  # Class 0
    ...     [0.0, 1.0, 0.0]   # Class 1
    ... ])  # Shape: (2, 3)
    >>> 
    >>> # Initialize CrossEntropyLoss with default reduction ("mean")
    >>> criterion = nn.CrossEntropyLoss()
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss)
    Tensor([[0.4076]], grad=None)  # Example loss value
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [[-0.0900,  0.0450,  0.0450],
     [ 0.0450, -0.0900,  0.0450]]  # Gradients with respect to input_tensor
    
**Using `CrossEntropyLoss` with "sum" reduction:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> import numpy as np
    >>> 
    >>> # Define input (logits) and target (one-hot) tensors
    >>> input_tensor = Tensor([
    ...     [1.0, 3.0, 0.2],
    ...     [2.0, 1.0, 0.1]
    ... ], requires_grad=True)  # Shape: (2, 3)
    >>> 
    >>> target_tensor = Tensor([
    ...     [0.0, 1.0, 0.0],  # Class 1
    ...     [1.0, 0.0, 0.0]   # Class 0
    ... ])  # Shape: (2, 3)
    >>> 
    >>> # Initialize CrossEntropyLoss with reduction="sum"
    >>> criterion = nn.CrossEntropyLoss(reduction="sum")
    >>> 
    >>> # Compute loss
    >>> loss = criterion(input_tensor, target_tensor)
    >>> print(loss)
    Tensor([[2.4076]], grad=None)  # Example loss value
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_tensor.grad)
    [[ 0.0000, -0.1800,  0.1800],
     [-0.1800,  0.0000,  0.1800]]  # Gradients with respect to input_tensor
    
**Using `CrossEntropyLoss` within a simple neural network:**
    
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
    ...         self.softmax = nn.Softmax(dim=1)
    ...     
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.softmax(x)
    ...         return x
    ...
    >>> 
    >>> # Initialize the model and loss function
    >>> model = SimpleClassificationModel(input_size=4, num_classes=3)
    >>> criterion = nn.CrossEntropyLoss()
    >>> 
    >>> # Define input and target tensors
    >>> input_data = Tensor([
    ...     [0.5, 1.2, -0.3, 2.0],
    ...     [1.5, -0.5, 2.3, 0.7]
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
        [0.1003, 0.8005, 0.0992],
        [0.7001, 0.1998, 0.1001]
    ], grad=None)  # Example output after linear transformation and softmax activation
    >>> 
    >>> # Compute loss
    >>> loss = criterion(output, target)
    >>> print(loss)
    Tensor([[0.2231]], grad=None)  # Example loss value
    >>> 
    >>> # Backpropagation
    >>> loss.backward()
    >>> print(input_data.grad)
    [
        [-0.0451,  0.0721,  0.0227, -0.0480],
        [ 0.0700, -0.0299, -0.0217,  0.0205]
    ]  # Gradients with respect to input_data
