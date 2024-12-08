nn.Conv2D
=========

.. autoclass:: lucid.nn.Conv2D
    
The `Conv2D` module performs a two-dimensional convolution operation over an input 
signal composed of several input channels. This layer is widely used in neural 
networks for tasks such as image recognition, computer vision, and spatial data 
analysis. The convolution operation allows the model to capture spatial hierarchies 
and patterns within the input data by applying learnable filters.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Conv2D(
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
    Number of channels in the input image.

- **out_channels** (*int*): 
    Number of channels produced by the convolution.

- **kernel_size** (*int* or *tuple[int, ...]*): 
    Size of the convolving kernel. Can be a single integer or a tuple specifying the size 
    in each spatial dimension.

- **stride** (*int* or *tuple[int, ...]*, optional): 
    Stride of the convolution. Default is `1`. Can be a single integer or a tuple specifying 
    the stride in each spatial dimension.

- **padding** (*_PaddingStr* or *int* or *tuple[int, ...]*, optional): 
    Zero-padding added to both sides of the input. Default is `0`. Can be a string (`'same'`, 
    `'valid'`) or an integer or a tuple specifying the padding in each spatial dimension.

- **dilation** (*int* or *tuple[int, ...]*, optional): 
    Spacing between kernel elements. Default is `1`. Can be a single integer or a tuple specifying 
    the dilation in each spatial dimension.

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
The `Conv2D` module computes the convolution of the input tensor with the weight tensor 
and adds the bias if enabled.

.. math::

    \mathbf{y} = \mathbf{x} * \mathbf{W} + \mathbf{b}

Where:

- :math:`\mathbf{x}` is the input tensor of shape `(N, C_{in}, H_{in}, W_{in})`.
- :math:`\mathbf{W}` is the weight tensor of shape `(C_{out}, C_{in} / \text{groups}, K_H, K_W)`.
- :math:`\mathbf{b}` is the bias tensor of shape `(C_{out})`, if applicable.
- :math:`\mathbf{y}` is the output tensor of shape `(N, C_{out}, H_{out}, W_{out})`.
- :math:`*` denotes the convolution operation.
- :math:`N` is the batch size.
- :math:`C_{in}` is the number of input channels.
- :math:`C_{out}` is the number of output channels.
- :math:`H_{in}`, :math:`W_{in}` are the height and width of the input.
- :math:`K_H`, :math:`K_W` are the height and width of the kernel.
- :math:`H_{out}`, :math:`W_{out}` are the height and width of the output, determined by 
  the convolution parameters.

Backward Gradient Calculation
-----------------------------
For tensors **x**, **W**, and **b** involved in the `Conv2D` operation, the gradients with 
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

This implies that during backpropagation, gradients flow through the weights and biases 
according to these derivatives, allowing the model to learn the optimal parameters.

Examples
--------
**Using `Conv2D` for a simple convolution without bias:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[
    ...     [[1.0, 2.0, 3.0],
    ...      [4.0, 5.0, 6.0],
    ...      [7.0, 8.0, 9.0]]
    ... ]], requires_grad=True)  # Shape: (1, 1, 3, 3)
    >>> conv2d = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    >>> print(conv2d.weight)
    Tensor([[
        [[1.0, 2.0],
         [3.0, 4.0]]
    ]], requires_grad=True)  # Shape: (1, 1, 2, 2)
    >>> output = conv2d(input_tensor)  # Shape: (1, 1, 2, 2)
    >>> print(output)
    Tensor([[
        [[1*1 + 2*2 + 4*3 + 5*4, 2*1 + 3*2 + 5*3 + 6*4],
         [4*1 + 5*2 + 7*3 + 8*4, 5*1 + 6*2 + 8*3 + 9*4]]
    ]], grad=None)  # Tensor([[[[27.0, 34.0],
                                [47.0, 54.0]]]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[
        [[1.0, 2.0],
         [3.0, 4.0]]
    ]])  # Gradient with respect to input_tensor
    >>> print(conv2d.weight.grad)
    Tensor([[
        [[1.0, 2.0],
         [4.0, 5.0]]
    ]])  # Gradient with respect to weight

**Using `Conv2D` with bias for a batch of inputs:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([
    ...     [
    ...         [[1.0, 2.0, 3.0, 4.0],
    ...          [5.0, 6.0, 7.0, 8.0],
    ...          [9.0, 10.0, 11.0, 12.0]]
    ...     ],
    ...     [
    ...         [[13.0, 14.0, 15.0, 16.0],
    ...          [17.0, 18.0, 19.0, 20.0],
    ...          [21.0, 22.0, 23.0, 24.0]]
    ...     ]
    ... ], requires_grad=True)  # Shape: (2, 1, 3, 4)
    >>> conv2d = nn.Conv2D(in_channels=1, out_channels=2, kernel_size=2, bias=True)
    >>> print(conv2d.weight)
    Tensor([
        [[[1.0, 2.0],
          [3.0, 4.0]]],
        [[[5.0, 6.0],
          [7.0, 8.0]]]
    ], requires_grad=True)  # Shape: (2, 1, 2, 2)
    >>> print(conv2d.bias)
    Tensor([9.0, 10.0], requires_grad=True)  # Shape: (2,)
    >>> output = conv2d(input_tensor)  # Shape: (2, 2, 2, 3)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([
        [
            [[1.0, 2.0],
             [3.0, 4.0]]
        ],
        [
            [[5.0, 6.0],
             [7.0, 8.0]]
        ]
    ])  # Gradients with respect to input_tensor
    >>> print(conv2d.weight.grad)
    Tensor([
        [
            [[1.0, 2.0],
             [3.0, 4.0]]
        ],
        [
            [[1.0, 2.0],
             [3.0, 4.0]]
        ]
    ])  # Gradients with respect to weight
    >>> print(conv2d.bias.grad)
    Tensor([2.0, 2.0])  # Gradients with respect to bias

**Integrating `Conv2D` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class Conv2DModel(nn.Module):
    ...     def __init__(self):
    ...         super(Conv2DModel, self).__init__()
    ...         self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    ...         self.relu = nn.ReLU()
    ...         self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    ...         self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
    ...         self.fc = nn.Linear(in_features=32 * 16 * 16, out_features=10)
    ...
    ...     def forward(self, x):
    ...         x = self.conv1(x)
    ...         x = self.relu(x)
    ...         x = self.conv2(x)
    ...         x = self.relu(x)
    ...         x = self.pool(x)
    ...         x = x.view(x.size(0), -1)
    ...         x = self.fc(x)
    ...         return x
    >>>
    >>> model = Conv2DModel()
    >>> input_data = Tensor([
    ...     [
    ...         [[0.5, -1.2, 3.3, 0.7],
    ...          [1.5, 2.2, -0.3, 4.1],
    ...          [5.5, 6.6, 7.7, 8.8],
    ...          [9.9, 10.1, 11.2, 12.3]]
    ...     ]], requires_grad=True)  # Shape: (1, 3, 4, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...]], grad=None)  # Output tensor after passing through the model
    
    # Backpropagation
    >>> output.backward()
