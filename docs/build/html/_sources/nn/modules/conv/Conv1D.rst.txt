nn.Conv1D
=========

.. autoclass:: lucid.nn.Conv1D

The `Conv1D` module performs a one-dimensional convolution operation over an input 
signal composed of several input channels. This layer is widely used in neural 
networks for tasks such as time series analysis, natural language processing, and 
audio processing. The convolution operation allows the model to capture local 
patterns and temporal dependencies within the input data.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Conv1D(
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
    in each dimension.

- **stride** (*int* or *tuple[int, ...]*, optional): 
    Stride of the convolution. Default is `1`. Can be a single integer or a tuple specifying 
    the stride in each dimension.

- **padding** (*_PaddingStr* or *int* or *tuple[int, ...]*, optional): 
    Zero-padding added to both sides of the input. Default is `0`. Can be a string (`'same'`, 
    `'valid'`) or an integer or a tuple specifying the padding in each dimension.

- **dilation** (*int* or *tuple[int, ...]*, optional): 
    Spacing between kernel elements. Default is `1`. Can be a single integer or a tuple specifying 
    the dilation in each dimension.

- **groups** (*int*, optional): 
    Number of blocked connections from input channels to output channels. Default is `1`.

- **bias** (*bool*, optional): 
    If `True`, adds a learnable bias to the output. Default is `True`.

Attributes
----------
- **weight** (*Tensor*): 
    Learnable weights of the module of shape `(out_channels, in_channels // groups, kernel_size)`. 
    Initialized from a uniform distribution.

- **bias** (*Tensor* or *None*): 
    Learnable bias of the module of shape `(out_channels)`. If `bias` is set to `False`, this 
    attribute is `None`.

Forward Calculation
-------------------
The `Conv1D` module computes the convolution of the input tensor with the weight tensor and 
adds the bias if enabled.

.. math::

    \mathbf{y} = \mathbf{x} * \mathbf{W} + \mathbf{b}

Where:

- :math:`\mathbf{x}` is the input tensor of shape `(N, C_{in}, L_{in})`.
- :math:`\mathbf{W}` is the weight tensor of shape `(C_{out}, C_{in} / \text{groups}, K)`.
- :math:`\mathbf{b}` is the bias tensor of shape `(C_{out})`, if applicable.
- :math:`\mathbf{y}` is the output tensor of shape `(N, C_{out}, L_{out})`.
- :math:`*` denotes the convolution operation.
- :math:`N` is the batch size.
- :math:`C_{in}` is the number of input channels.
- :math:`C_{out}` is the number of output channels.
- :math:`L_{in}` is the length of the input signal.
- :math:`L_{out}` is the length of the output signal, determined by the convolution parameters.

Backward Gradient Calculation
-----------------------------
For tensors **x**, **W**, and **b** involved in the `Conv1D` operation, the gradients with 
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
**Using `Conv1D` for a simple convolution without bias:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[[1.0, 2.0, 3.0, 4.0]]], requires_grad=True)  # Shape: (1, 1, 4)
    >>> conv1d = nn.Conv1D(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    >>> print(conv1d.weight)
    Tensor([[[5.0, 6.0]]], requires_grad=True)  # Shape: (1, 1, 2)
    >>> output = conv1d(input_tensor)  # Shape: (1, 1, 3)
    >>> print(output)
    Tensor([[[1*5 + 2*6, 2*5 + 3*6, 3*5 + 4*6]]], grad=None)  # Tensor([[ [17.0, 27.0, 37.0]]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[[5.0, 6.0, 5.0]]])  # Gradient with respect to input_tensor
    >>> print(conv1d.weight.grad)
    Tensor([[[1.0, 2.0, 3.0]]])  # Gradient with respect to weight

**Using `Conv1D` with bias for a batch of inputs:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([
    ...     [[1.0, 2.0, 3.0, 4.0, 5.0]],
    ...     [[6.0, 7.0, 8.0, 9.0, 10.0]]
    ... ], requires_grad=True)  # Shape: (2, 1, 5)
    >>> conv1d = nn.Conv1D(in_channels=1, out_channels=2, kernel_size=3, bias=True)
    >>> print(conv1d.weight)
    Tensor([
        [[11.0, 12.0, 13.0]],
        [[14.0, 15.0, 16.0]]
    ], requires_grad=True)  # Shape: (2, 1, 3)
    >>> print(conv1d.bias)
    Tensor([17.0, 18.0], requires_grad=True)  # Shape: (2,)
    >>> output = conv1d(input_tensor)  # Shape: (2, 2, 3)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([
        [[11.0, 12.0, 13.0],
         [14.0, 15.0, 16.0]],
        [[11.0, 12.0, 13.0],
         [14.0, 15.0, 16.0]]
    ])  # Gradients with respect to input_tensor
    >>> print(conv1d.weight.grad)
    Tensor([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
    ])  # Gradients with respect to weight
    >>> print(conv1d.bias.grad)
    Tensor([2.0, 2.0])  # Gradients with respect to bias

**Integrating `Conv1D` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class Conv1DModel(nn.Module):
    ...     def __init__(self):
    ...         super(Conv1DModel, self).__init__()
    ...         self.conv1 = nn.Conv1D(in_channels=1, out_channels=4, kernel_size=3, padding=1)
    ...         self.relu = nn.ReLU()
    ...         self.conv2 = nn.Conv1D(in_channels=4, out_channels=2, kernel_size=3)
    ...
    ...     def forward(self, x):
    ...         x = self.conv1(x)
    ...         x = self.relu(x)
    ...         x = self.conv2(x)
    ...         return x
    >>>
    >>> model = Conv1DModel()
    >>> input_data = Tensor([
    ...     [[0.5, -1.2, 3.3, 0.7, 2.2]],
    ...     [[1.5, 2.2, -0.3, 4.1, 5.5]]
    ... ], requires_grad=True)  # Shape: (2, 1, 5)
    >>> output = model(input_data)
    
    # Backpropagation
    >>> output.backward()
