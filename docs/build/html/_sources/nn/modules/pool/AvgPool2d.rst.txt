nn.AvgPool2d
============

.. autoclass:: lucid.nn.AvgPool2d
    
The `AvgPool2d` module applies a two-dimensional average pooling operation over an input 
signal composed of several input channels. This layer is commonly used in convolutional 
neural networks to reduce the spatial dimensions (height and width) of the input, thereby 
reducing the number of parameters and computation in the network. 

Average pooling summarizes the features present in patches of the input by computing the 
average value within each window, helping to make the representation approximately invariant 
to small translations of the input.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.AvgPool2d(
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0
    )

Parameters
----------
- **kernel_size** (*int* or *tuple[int, ...]*, optional):
    Size of the window to take an average over. Can be a single integer or a tuple specifying 
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
The `AvgPool2d` module performs the following operation:

.. math::

    \mathbf{y}_{i,j} = \frac{1}{k_h k_w} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} 
    \mathbf{x}_{i \times s_h + m - p_h, j \times s_w + n - p_w}

Where:

- :math:`\mathbf{x}` is the input tensor of shape `(N, C, H_{in}, W_{in})`.
- :math:`\mathbf{y}` is the output tensor of shape `(N, C, H_{out}, W_{out})`.
- :math:`k_h` and :math:`k_w` are the kernel heights and widths.
- :math:`s_h` and :math:`s_w` are the strides for height and width.
- :math:`p_h` and :math:`p_w` are the padding for height and width.
- :math:`N` is the batch size.
- :math:`C` is the number of channels.
- :math:`H_{in}`, :math:`W_{in}` are the height and width of the input.
- :math:`H_{out}`, :math:`W_{out}` are the height and width of the output, determined by 
  the pooling parameters.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed by distributing 
the gradient from each output element equally to all elements in the corresponding pooling window.

.. math::

    \frac{\partial \mathbf{y}_{i,j}}{\partial \mathbf{x}_{m,n}} =
    \begin{cases}
        \frac{1}{k_h k_w} & \text{if } (m,n) \text{ is within the pooling window for } (i,j) \\
        0 & \text{otherwise}
    \end{cases}

Where:

- :math:`\mathbf{y}_{i,j}` is the output at position `(i, j)`.
- :math:`\mathbf{x}_{m,n}` is the input at position `(m, n)`.

This ensures that the gradient is appropriately averaged and propagated back to the input tensor.

Examples
--------
**Using `AvgPool2d` with a simple input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[
    ...     [[1.0, 2.0, 3.0, 4.0],
    ...      [5.0, 6.0, 7.0, 8.0],
    ...      [9.0, 10.0, 11.0, 12.0],
    ...      [13.0, 14.0, 15.0, 16.0]]
    ... ]], requires_grad=True)  # Shape: (1, 1, 4, 4)
    >>> avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    >>> output = avg_pool(input_tensor)  # Shape: (1, 1, 2, 2)
    >>> print(output)
    Tensor([[
        [[3.5, 5.5],
         [11.5, 13.5]]
    ]], grad=None)

    # Backpropagation
    >>> output.backward(Tensor([[
    ...     [[1.0, 1.0],
    ...      [1.0, 1.0]]
    ... ]]))
    >>> print(input_tensor.grad)
    Tensor([[
        [[0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]]
    ]])  # Gradients with respect to input_tensor

**Using `AvgPool2d` with padding:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[
    ...     [[1.0, 2.0, 3.0],
    ...      [4.0, 5.0, 6.0],
    ...      [7.0, 8.0, 9.0]]
    ... ]], requires_grad=True)  # Shape: (1, 1, 3, 3)
    >>> avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
    >>> output = avg_pool(input_tensor)  # Shape: (1, 1, 2, 2)
    >>> print(output)
    Tensor([[
        [[1.5, 3.5],
         [4.5, 6.5]]
    ]], grad=None)

    # Backpropagation
    >>> output.backward(Tensor([[
    ...     [[1.0, 1.0],
    ...      [1.0, 1.0]]
    ... ]]))
    >>> print(input_tensor.grad)
    Tensor([[
        [[0.25, 0.5, 0.25],
         [0.5, 1.0, 0.5],
         [0.25, 0.5, 0.25]]
    ]])  # Gradients with respect to input_tensor

**Integrating `AvgPool2d` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class AvgPool2dModel(nn.Module):
    ...     def __init__(self):
    ...         super(AvgPool2dModel, self).__init__()
    ...         self.conv1 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
    ...         self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    ...         self.fc = nn.Linear(in_features=1 * 2 * 2, out_features=1)
    ...
    ...     def forward(self, x):
    ...         x = self.conv1(x)
    ...         x = self.avg_pool(x)
    ...         x = x.view(x.size(0), -1)
    ...         x = self.fc(x)
    ...         return x
    ...
    >>> model = AvgPool2dModel()
    >>> input_data = Tensor([[
    ...     [[1.0, 2.0, 3.0, 4.0],
    ...      [5.0, 6.0, 7.0, 8.0],
    ...      [9.0, 10.0, 11.0, 12.0],
    ...      [13.0, 14.0, 15.0, 16.0]]
    ... ]], requires_grad=True)  # Shape: (1, 1, 4, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[[...]]], grad=None)  # Output tensor after passing through the model

    # Backpropagation
    >>> output.backward(Tensor([[[1.0]]]))
    >>> print(input_data.grad)
    Tensor([[
        [[0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]]
    ]])  # Gradients with respect to input_data
