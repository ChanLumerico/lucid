nn.MaxPool2d
============

.. autoclass:: lucid.nn.MaxPool2d

The `MaxPool2d` module applies a two-dimensional maximum pooling operation over an input 
signal composed of several input channels. This layer is commonly used in convolutional 
neural networks to reduce the spatial dimensions (height and width) of the input, thereby 
reducing the number of parameters and computation in the network. 

The maximum pooling operation highlights the most prominent features within each window, 
helping to make the representation approximately invariant to small translations of the input.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.MaxPool2d(
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
The `MaxPool2d` module performs the following operation:

.. math::

    \mathbf{y}_{i,j} = \max \left( \mathbf{x}_{i \times s + m - p,\ j \times s + n - p} \right) 
    \quad \text{for} \quad m = 0, \dots, k-1 \text{ and } n = 0, \dots, k-1

Where:

- :math:`\mathbf{x}` is the input tensor of shape :math:`(N, C, H_{in}, W_{in})`.
- :math:`\mathbf{y}` is the output tensor of shape :math:`(N, C, H_{out}, W_{out})`.
- :math:`k` is the `kernel_size`.
- :math:`s` is the `stride`.
- :math:`p` is the `padding`.
- :math:`N` is the batch size.
- :math:`C` is the number of channels.
- :math:`H_{in}`, :math:`W_{in}` are the height and width of the input.
- :math:`H_{out}`, :math:`W_{out}` are the height and width of the output, determined by 
  the pooling parameters.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is routed to the position of the 
maximum value in each pooling window, and zero elsewhere.

.. math::

    \frac{\partial \mathbf{y}_{i,j}}{\partial \mathbf{x}_{m,n}} =
    \begin{cases}
        1 & \text{if } \mathbf{x}_{m,n} \text{ is the max in its pooling window} \\
        0 & \text{otherwise}
    \end{cases}

This ensures that only the input element contributing to the maximum value receives the gradient.

Examples
--------
**Using `MaxPool2d` with a simple input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[
    ...     [[1.0, 3.0],
    ...      [2.0, 4.0]]
    ... ]], requires_grad=True)  # Shape: (1, 1, 2, 2)
    >>> max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    >>> output = max_pool(input_tensor)  # Shape: (1, 1, 1, 1)
    >>> print(output)
    Tensor([[[[4.0]]]], grad=None)
    
    # Backpropagation
    >>> output.backward(Tensor([[[[1.0]]]]))
    >>> print(input_tensor.grad)
    Tensor([[
        [[0.0, 0.0],
         [0.0, 1.0]]
    ]])  # Gradients with respect to input_tensor

**Using `MaxPool2d` with padding:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[
    ...     [[1.0, 2.0, 3.0],
    ...      [4.0, 5.0, 6.0],
    ...      [7.0, 8.0, 9.0]]
    ... ]], requires_grad=True)  # Shape: (1, 1, 3, 3)
    >>> max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
    >>> output = max_pool(input_tensor)  # Shape: (1, 1, 3, 3)
    >>> print(output)
    Tensor([[
        [[1.0, 3.0, 3.0],
         [5.0, 6.0, 6.0],
         [7.0, 9.0, 9.0]]
    ]], grad=None)
    
    # Backpropagation
    >>> output.backward(Tensor([[
    ...     [[1.0, 1.0, 1.0],
    ...      [1.0, 1.0, 1.0],
    ...      [1.0, 1.0, 1.0]]
    ... ]]))
    >>> print(input_tensor.grad)
    Tensor([[
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]
    ]])  # Gradients with respect to input_tensor

**Integrating `MaxPool2d` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class MaxPool2dModel(nn.Module):
    ...     def __init__(self):
    ...         super(MaxPool2dModel, self).__init__()
    ...         self.conv1 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
    ...         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    ...         self.fc = nn.Linear(in_features=1 * 1 * 1, out_features=1)
    ...
    ...     def forward(self, x):
    ...         x = self.conv1(x)
    ...         x = self.max_pool(x)
    ...         x = x.view(x.size(0), -1)
    ...         x = self.fc(x)
    ...         return x
    ...
    >>> model = MaxPool2dModel()
    >>> input_data = Tensor([[
    ...     [[1.0, 2.0, 3.0, 4.0],
    ...      [5.0, 6.0, 7.0, 8.0],
    ...      [9.0, 10.0, 11.0, 12.0],
    ...      [13.0, 14.0, 15.0, 16.0]]
    ... ]], requires_grad=True)  # Shape: (1, 1, 4, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[[16.0]]], grad=None)  # Example output after passing through the model
    
    # Backpropagation
    >>> output.backward(Tensor([[[1.0]]]))
    >>> print(input_data.grad)
    Tensor([[
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]]
    ]])  # Gradients with respect to input_data
