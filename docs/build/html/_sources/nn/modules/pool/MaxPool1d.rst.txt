nn.MaxPool1d
============

.. autoclass:: lucid.nn.MaxPool1d
    
The `MaxPool1d` module applies a one-dimensional maximum pooling operation over an 
input signal composed of several input channels. This layer is commonly used in neural 
networks for tasks such as time series analysis and natural language processing. 

The maximum pooling operation reduces the dimensionality of the input by selecting the 
maximum value within sliding windows, helping to highlight prominent features and reduce 
computational complexity.
    
Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.MaxPool1d(
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0
    )
    
Parameters
----------
- **kernel_size** (*int* or *tuple[int, ...]*, optional):
    Size of the window to take a maximum over. Can be a single integer or a tuple specifying 
    the size in each spatial dimension. Default is `1`.
    
- **stride** (*int* or *tuple[int, ...]*, optional):
    Stride of the window. Can be a single integer or a tuple specifying the stride in each 
    spatial dimension. If not provided, it defaults to the same value as `kernel_size`. 
    Default is `1`.
    
- **padding** (*int* or *tuple[int, ...]*, optional):
    Zero-padding added to both sides of the input. Can be a single integer or a tuple 
    specifying the padding in each spatial dimension. Default is `0`.
    
Attributes
----------
- **None**
    
Forward Calculation
-------------------
The `MaxPool1d` module performs the following operation:
    
.. math::
    
    \mathbf{y}_i = \max \left( \mathbf{x}_{i \times s + j - p} \right) \quad \text{for} 
    \quad j = 0, \dots, k-1
    
Where:

- :math:`\mathbf{x}` is the input tensor of shape :math:`(N, C, L_{in})`.
- :math:`\mathbf{y}` is the output tensor of shape :math:`(N, C, L_{out})`.
- :math:`k` is the `kernel_size`.
- :math:`s` is the `stride`.
- :math:`p` is the `padding`.
- :math:`N` is the batch size.
- :math:`C` is the number of channels.
- :math:`L_{in}` and :math:`L_{out}` are the lengths of the input and output signals, respectively.
    
Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is routed to the position of the 
maximum value in each pooling window, and zero elsewhere.
    
.. math::
    
    \frac{\partial \mathbf{y}_i}{\partial \mathbf{x}_j} =
    \begin{cases}
        1 & \text{if } \mathbf{x}_j \text{ is the max in its pooling window} \\
        0 & \text{otherwise}
    \end{cases}
    
This ensures that only the input element contributing to the maximum value receives the gradient.

Examples
--------
**Using `MaxPool1d` with a simple input tensor:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[[1.0, 3.0, 2.0, 4.0]]], requires_grad=True)  # Shape: (1, 1, 4)
    >>> max_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    >>> output = max_pool(input_tensor)  # Shape: (1, 1, 2)
    >>> print(output)
    Tensor([[[3.0, 4.0]]], grad=None)
    
    # Backpropagation
    >>> output.backward(Tensor([[[1.0, 1.0]]]))
    >>> print(input_tensor.grad)
    Tensor([[[0.0, 1.0, 0.0, 1.0]]])  # Gradients with respect to input_tensor
    
**Using `MaxPool1d` with padding:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)  # Shape: (1, 1, 3)
    >>> max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
    >>> output = max_pool(input_tensor)  # Shape: (1, 1, 2)
    >>> print(output)
    Tensor([[[2.0, 3.0]]], grad=None)
    
    # Backpropagation
    >>> output.backward(Tensor([[[1.0, 1.0]]]))
    >>> print(input_tensor.grad)
    Tensor([[[0.0, 1.0, 1.0]]])  # Gradients with respect to input_tensor
    
**Integrating `MaxPool1d` into a Neural Network Model:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class MaxPool1dModel(nn.Module):
    ...     def __init__(self):
    ...         super(MaxPool1dModel, self).__init__()
    ...         self.conv1 = nn.Conv1D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
    ...         self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    ...         self.fc = nn.Linear(in_features=1 * 2, out_features=1)
    ...
    ...     def forward(self, x):
    ...         x = self.conv1(x)
    ...         x = self.max_pool(x)
    ...         x = x.view(x.size(0), -1)
    ...         x = self.fc(x)
    ...         return x
    ...
    >>> model = MaxPool1dModel()
    >>> input_data = Tensor([[[1.0, 2.0, 3.0, 4.0]]], requires_grad=True)  # Shape: (1, 1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[[...]]], grad=None)  # Output tensor after passing through the model
    
    # Backpropagation
    >>> output.backward(Tensor([[[1.0]]]))
    >>> print(input_data.grad)
    Tensor([[[0.0, 1.0, 0.0, 1.0]]])  # Gradients with respect to input_data
