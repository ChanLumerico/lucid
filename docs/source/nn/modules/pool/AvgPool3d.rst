nn.AvgPool3d
============

.. autoclass:: lucid.nn.AvgPool3d
    
The `AvgPool3d` module applies a three-dimensional average pooling operation over an input 
signal composed of several input channels. This layer is commonly used in convolutional 
neural networks for tasks such as video processing and volumetric data analysis. The 
average pooling operation reduces the spatial and temporal dimensions of the input by 
computing the average value within sliding windows, thereby summarizing and extracting 
prominent features from the input data.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.AvgPool3d(
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
    Zero-padding added to all sides of the input. Can be a single integer or a tuple 
    specifying the padding in each spatial dimension. Default is `0`.

Attributes
----------
- **None**

Forward Calculation
-------------------
The `AvgPool3d` module performs the following operation:

.. math::

    \mathbf{y}_{d,h,w} = \frac{1}{k_d k_h k_w} \sum_{m=0}^{k_d-1} 
    \sum_{n=0}^{k_h-1} \sum_{o=0}^{k_w-1} \mathbf{x}_{d \times s_d + m - p_d,\ h 
    \times s_h + n - p_h,\ w \times s_w + o - p_w}

Where:

- :math:`\mathbf{x}` is the input tensor of shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
- :math:`\mathbf{y}` is the output tensor of shape :math:`(N, C, D_{out}, H_{out}, W_{out})`.
- :math:`k_d`, :math:`k_h`, :math:`k_w` are the kernel sizes for depth, height, and width.
- :math:`s_d`, :math:`s_h`, :math:`s_w` are the strides for depth, height, and width.
- :math:`p_d`, :math:`p_h`, :math:`p_w` are the padding for depth, height, and width.
- :math:`N` is the batch size.
- :math:`C` is the number of channels.
- :math:`D_{in}`, :math:`H_{in}`, :math:`W_{in}` are the depth, height, and width of the input.
- :math:`D_{out}`, :math:`H_{out}`, :math:`W_{out}` are the depth, height, and width of the output, 
  determined by the pooling parameters.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed by distributing 
the gradient from each output element equally to all elements in the corresponding pooling window.

.. math::

    \frac{\partial \mathbf{y}_{d,h,w}}{\partial \mathbf{x}_{m,n,o}} =
    \begin{cases}
        \frac{1}{k_d k_h k_w} & \text{if } (m,n,o) \text{ is within the pooling window for } (d,h,w) \\\
        0 & \text{otherwise}
    \end{cases}

Where:

- :math:`\mathbf{y}_{d,h,w}` is the output at position :math:`(d, h, w)`.
- :math:`\mathbf{x}_{m,n,o}` is the input at position :math:`(m, n, o)`.

This ensures that the gradient is appropriately averaged and propagated back to the input tensor.

Examples
--------
**Using `AvgPool3d` with a simple input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[
    ...     [[1.0, 2.0],
    ...      [3.0, 4.0]],
    ...     [[5.0, 6.0],
    ...      [7.0, 8.0]]
    ... ]], requires_grad=True)  # Shape: (1, 2, 2, 2, 2)
    >>> avg_pool = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
    >>> output = avg_pool(input_tensor)  # Shape: (1, 2, 1, 1, 1)
    >>> print(output)
    Tensor([[
        [[[3.0]]],
        [[[7.0]]]
    ]], grad=None)

    # Backpropagation
    >>> output.backward(Tensor([[
    ...     [[[1.0]]],
    ...     [[[1.0]]]
    ... ]]))
    >>> print(input_tensor.grad)
    Tensor([[
        [[[0.125, 0.125],
          [0.125, 0.125]],
        [[[0.125, 0.125],
          [0.125, 0.125]]]
    ]])  # Gradients with respect to input_tensor

**Using `AvgPool3d` with padding:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[
    ...     [[1.0, 2.0, 3.0],
    ...      [4.0, 5.0, 6.0],
    ...      [7.0, 8.0, 9.0]],
    ...     [[10.0, 11.0, 12.0],
    ...      [13.0, 14.0, 15.0],
    ...      [16.0, 17.0, 18.0]]
    ... ]], requires_grad=True)  # Shape: (1, 2, 3, 3, 3)
    >>> avg_pool = nn.AvgPool3d(kernel_size=2, stride=1, padding=1)
    >>> output = avg_pool(input_tensor)  # Shape: (1, 2, 2, 2, 2)
    >>> print(output)
    Tensor([
        [[[3.0, 4.0],
          [6.0, 7.0]],
        [[[9.0, 10.0],
          [12.0, 13.0]]]
    ]], grad=None)

    # Backpropagation
    >>> output.backward(Tensor([
    ...     [[[1.0, 1.0],
    ...       [1.0, 1.0]],
    ...     [[[1.0, 1.0],
    ...       [1.0, 1.0]]]
    ... ]]))
    >>> print(input_tensor.grad)
    Tensor([
        [[[0.125, 0.25, 0.125],
          [0.25, 0.5, 0.25],
          [0.125, 0.25, 0.125]],
        [[[0.125, 0.25, 0.125],
          [0.25, 0.5, 0.25],
          [0.125, 0.25, 0.125]]]
    ]])  # Gradients with respect to input_tensor

**Integrating `AvgPool3d` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class AvgPool3dModel(nn.Module):
    ...     def __init__(self):
    ...         super(AvgPool3dModel, self).__init__()
    ...         self.conv1 = nn.Conv3D(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
    ...         self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
    ...         self.fc = nn.Linear(in_features=4 * 1 * 1 * 1, out_features=1)
    ...
    ...     def forward(self, x):
    ...         x = self.conv1(x)
    ...         x = self.avg_pool(x)
    ...         x = x.view(x.size(0), -1)
    ...         x = self.fc(x)
    ...         return x
    ...
    >>> model = AvgPool3dModel()
    >>> input_data = Tensor([
    ...     [[[1.0, 2.0, 3.0],
    ...       [4.0, 5.0, 6.0],
    ...       [7.0, 8.0, 9.0]],
    ...      [[10.0, 11.0, 12.0],
    ...       [13.0, 14.0, 15.0],
    ...       [16.0, 17.0, 18.0]]]
    ... ], requires_grad=True)  # Shape: (1, 2, 3, 3, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[[...]]], grad=None)  # Output tensor after passing through the model

    # Backpropagation
    >>> output.backward(Tensor([[[1.0]]]))
    >>> print(input_data.grad)
    Tensor([
        [[[0.125, 0.25, 0.125],
          [0.25, 0.5, 0.25],
          [0.125, 0.25, 0.125]],
        [[[0.125, 0.25, 0.125],
          [0.25, 0.5, 0.25],
          [0.125, 0.25, 0.125]]]
    ]])  # Gradients with respect to input_data
