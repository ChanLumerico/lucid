nn.Conv3D
=========

.. autoclass:: lucid.nn.Conv3D
    
The `Conv3D` module performs a three-dimensional convolution operation over an input 
signal composed of several input channels. This layer is commonly used in neural 
networks for tasks such as video analysis, volumetric data processing, and 
spatio-temporal feature extraction. The convolution operation enables the model 
to capture spatial and temporal patterns within the input data by applying 
learnable 3D filters.
    
Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.Conv3D(
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True
    )
    
Parameters
----------
- **in_channels** (*int*): 
    Number of channels in the input signal.
    
- **out_channels** (*int*): 
    Number of channels produced by the convolution.
    
- **kernel_size** (*int* or *tuple[int, ...]*): 
    Size of the convolving kernel. Can be a single integer or a tuple specifying the size 
    in each spatial dimension.
    
- **stride** (*int* or *tuple[int, ...]*, optional): 
    Stride of the convolution. Default is `1`. Can be a single integer or a tuple specifying 
    the stride in each spatial dimension.
    
- **padding** (*_PaddingStr* or *int* or *tuple[int, ...]*, optional): 
    Zero-padding added to all sides of the input. Default is `0`. Can be a string 
    (`'same'`, `'valid'`), an integer, or a tuple specifying the padding in each 
    spatial dimension.
    
- **dilation** (*int* or *tuple[int, ...]*, optional): 
    Spacing between kernel elements. Default is `1`. Can be a single integer or a tuple 
    specifying the dilation in each spatial dimension.
    
- **groups** (*int*, optional): 
    Number of blocked connections from input channels to output channels. Default is `1`.
    
- **bias** (*bool*, optional): 
    If `True`, adds a learnable bias to the output. Default is `True`.
    
Attributes
----------
- **weight** (*Tensor*): 
    Learnable weights of the module of shape `(out_channels, in_channels // groups, *kernel_size)`. 
    Initialized from a uniform distribution.
    
- **bias** (*Tensor* or *None*): 
    Learnable bias of the module of shape `(out_channels)`. If `bias` is set to `False`, this 
    attribute is `None`.
    
Forward Calculation
-------------------
The `Conv3D` module computes the convolution of the input tensor with the weight tensor 
and adds the bias if enabled.
    
.. math::
    
    \mathbf{y} = \mathbf{x} * \mathbf{W} + \mathbf{b}
    
Where:
    
- :math:`\mathbf{x}` is the input tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.
- :math:`\mathbf{W}` is the weight tensor of shape :math:`(C_{out}, C_{in} / \text{groups}, K_D, K_H, K_W)`.
- :math:`\mathbf{b}` is the bias tensor of shape :math:`(C_{out})`, if applicable.
- :math:`\mathbf{y}` is the output tensor of shape :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`.
- :math:`*` denotes the convolution operation.
- :math:`N` is the batch size.
- :math:`C_{in}` is the number of input channels.
- :math:`C_{out}` is the number of output channels.
- :math:`D_{in}`, :math:`H_{in}`, :math:`W_{in}` are the depth, height, and width of the input.
- :math:`K_D`, :math:`K_H`, :math:`K_W` are the depth, height, and width of the kernel.
- :math:`D_{out}`, :math:`H_{out}`, :math:`W_{out}` are the depth, height, and width of the output, determined by 
  the convolution parameters.
    
Backward Gradient Calculation
-----------------------------
For tensors **x**, **W**, and **b** involved in the `Conv3D` operation, the gradients with 
respect to the output (**y**) are computed as follows:
    
**Gradient with respect to** :math:`\mathbf{x}`:
    
.. math::
    
    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{W}^\top * \mathbf{1}
    
**Gradient with respect to** :math:`\mathbf{W}`:
    
.. math::
    
    \frac{\partial \mathbf{y}}{\partial \mathbf{W}} = \mathbf{x} * \mathbf{1}^\top
    
**Gradient with respect to** :math:`\mathbf{b}` (if bias is used):
    
.. math::
    
    \frac{\partial \mathbf{y}}{\partial \mathbf{b}} = \mathbf{1}
    
Where:
    
- :math:`\mathbf{1}` represents a tensor of ones with appropriate shape to facilitate 
  gradient computation.
    
This means that during backpropagation, gradients flow through the weights and biases 
according to these derivatives, allowing the model to learn the optimal parameters.
    
Examples
--------
**Using `Conv3D` for a simple convolution without bias:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[
    ...     [[1.0, 2.0],
    ...      [3.0, 4.0]],
    ...     [[5.0, 6.0],
    ...      [7.0, 8.0]]
    ... ]], requires_grad=True)  # Shape: (1, 2, 2, 2, 2)
    >>> conv3d = nn.Conv3D(in_channels=2, out_channels=1, kernel_size=2, bias=False)
    >>> print(conv3d.weight)
    Tensor([[
        [[[1.0, 2.0],
          [3.0, 4.0]],
         [[5.0, 6.0],
          [7.0, 8.0]]]
    ]], requires_grad=True)  # Shape: (1, 2, 2, 2, 2)
    >>> output = conv3d(input_tensor)  # Shape: (1, 1, 1, 1, 1)
    >>> print(output)
    Tensor([[[[[204.0]]]]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[
        [[[1.0, 2.0],
          [3.0, 4.0]],
         [[5.0, 6.0],
          [7.0, 8.0]]]
    ]])  # Gradient with respect to input_tensor
    >>> print(conv3d.weight.grad)
    Tensor([[
        [[[1.0, 2.0],
          [3.0, 4.0]],
         [[1.0, 2.0],
          [3.0, 4.0]]]
    ]])  # Gradient with respect to weight
    
**Using `Conv3D` with bias for a batch of inputs:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([
    ...     [
    ...         [[1.0, 2.0, 3.0],
    ...          [4.0, 5.0, 6.0],
    ...          [7.0, 8.0, 9.0]],
    ...         [[10.0, 11.0, 12.0],
    ...          [13.0, 14.0, 15.0],
    ...          [16.0, 17.0, 18.0]]
    ...     ],
    ...     [
    ...         [[19.0, 20.0, 21.0],
    ...          [22.0, 23.0, 24.0],
    ...          [25.0, 26.0, 27.0]],
    ...         [[28.0, 29.0, 30.0],
    ...          [31.0, 32.0, 33.0],
    ...          [34.0, 35.0, 36.0]]
    ...     ]
    ... ], requires_grad=True)  # Shape: (2, 2, 3, 3, 3)
    >>> conv3d = nn.Conv3D(in_channels=2, out_channels=2, kernel_size=2, bias=True)
    >>> print(conv3d.weight)
    Tensor([
        [
            [[[1.0, 2.0],
              [3.0, 4.0]],
             [[5.0, 6.0],
              [7.0, 8.0]]]
        ],
        [
            [[[9.0, 10.0],
              [11.0, 12.0]],
             [[13.0, 14.0],
              [15.0, 16.0]]]
        ]
    ], requires_grad=True)  # Shape: (2, 2, 2, 2, 2)
    >>> print(conv3d.bias)
    Tensor([17.0, 18.0], requires_grad=True)  # Shape: (2,)
    >>> output = conv3d(input_tensor)  # Shape: (2, 2, 2, 2, 2)
    >>> print(output)
    Tensor([[
        [[[...]], [[...]]]
    ],
    [
        [[[...]], [[...]]]
    ]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    
**Integrating `Conv3D` into a Neural Network Model:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> class Conv3DModel(nn.Module):
    ...     def __init__(self):
    ...         super(Conv3DModel, self).__init__()
    ...         self.conv1 = nn.Conv3D(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    ...         self.relu = nn.ReLU()
    ...         self.conv2 = nn.Conv3D(in_channels=8, out_channels=16, kernel_size=3)
    ...         self.pool = nn.MaxPool3D(kernel_size=2, stride=2)
    ...         self.fc = nn.Linear(in_features=16 * 4 * 4 * 4, out_features=10)
    ...
    ...     def forward(self, x):
    ...         x = self.conv1(x)
    ...         x = self.relu(x)
    ...         x = self.conv2(x)
    ...         x = self.relu(x)
    ...         x = self.pool(x)
    ...         x = x.reshape(x.shape[0], -1)
    ...         x = self.fc(x)
    ...         return x
    >>>
    >>> model = Conv3DModel()
    >>> input_data = Tensor(
    ... [
    ...     [[[0.5, -1.2, 3.3],
    ...       [0.7, 2.2, -0.3],
    ...       [4.1, 5.5, 6.6]]]
    ... ], requires_grad=True)  # Shape: (1, 1, 3, 3, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...]], grad=None)  # Output tensor after passing through the model
    
    # Backpropagation
    >>> output.backward()
