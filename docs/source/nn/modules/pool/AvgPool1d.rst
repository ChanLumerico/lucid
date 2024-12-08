nn.AvgPool1d
============

.. autoclass:: lucid.nn.AvgPool1d

The `AvgPool1d` module applies a one-dimensional average pooling operation over an 
input signal composed of several input channels. This layer is commonly used in neural 
networks for tasks such as time series analysis and natural language processing. 

The average pooling operation reduces the dimensionality of the input by computing 
the average value within sliding windows, helping to summarize and extract prominent 
features from the input data.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.AvgPool1d(
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0
    )

Parameters
----------
- **kernel_size** (*int* or *tuple[int, ...]*, optional):
    Size of the window to take an average over. Default is `1`.

- **stride** (*int* or *tuple[int, ...]*, optional):
    Stride of the window. Default is `1`. If not provided, it defaults to the 
    same value as `kernel_size`.

- **padding** (*int* or *tuple[int, ...]*, optional):
    Zero-padding added to both sides of the input. Default is `0`.

Attributes
----------
- **None**

Forward Calculation
-------------------
The `AvgPool1d` module performs the following operation:

.. math::

    \mathbf{y}_i = \frac{1}{k} \sum_{j=0}^{k-1} \mathbf{x}_{i \times s + j - p}

Where:

- :math:`\mathbf{x}` is the input tensor of shape `(N, C, L_{in})`.
- :math:`\mathbf{y}` is the output tensor of shape `(N, C, L_{out})`.
- :math:`k` is the `kernel_size`.
- :math:`s` is the `stride`.
- :math:`p` is the `padding`.
- :math:`L_{in}` and :math:`L_{out}` are the lengths of the input and output signals, respectively.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed by 
distributing the gradient from the output equally to each element in the pooling window.

.. math::

    \frac{\partial \mathbf{y}_i}{\partial \mathbf{x}_j} =
    \begin{cases}
        \frac{1}{k} & \text{if } j \in \text{window}_i \\
        0 & \text{otherwise}
    \end{cases}

Where:

- :math:`\mathbf{y}_i` is the output at position `i`.
- :math:`\mathbf{x}_j` is the input at position `j`.
- :math:`\text{window}_i` defines the indices of the input that contribute 
  to the output at position `i`.

Examples
--------
**Using `AvgPool1d` with a simple input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[[1.0, 2.0, 3.0, 4.0]]], requires_grad=True)  # Shape: (1, 1, 4)
    >>> avg_pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
    >>> output = avg_pool(input_tensor)  # Shape: (1, 1, 2)
    >>> print(output)
    Tensor([[[1.5, 3.5]]], grad=None)

    # Backpropagation
    >>> output.backward(Tensor([[[1.0, 1.0]]]))
    >>> print(input_tensor.grad)
    Tensor([[[0.5, 0.5, 0.5, 0.5]]])  # Gradients with respect to input_tensor

**Using `AvgPool1d` with padding:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)  # Shape: (1, 1, 3)
    >>> avg_pool = nn.AvgPool1d(kernel_size=2, stride=1, padding=1)
    >>> output = avg_pool(input_tensor)  # Shape: (1, 1, 2)
    >>> print(output)
    Tensor([[[0.5, 2.0]]], grad=None)

    # Backpropagation
    >>> output.backward(Tensor([[[1.0, 1.0]]]))
    >>> print(input_tensor.grad)
    Tensor([[[0.5, 1.0, 0.5]]])  # Gradients with respect to input_tensor

**Integrating `AvgPool1d` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class AvgPool1dModel(nn.Module):
    ...     def __init__(self):
    ...         super(AvgPool1dModel, self).__init__()
    ...         self.conv1 = nn.Conv1D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
    ...         self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
    ...         self.fc = nn.Linear(in_features=1, out_features=1)
    ...
    ...     def forward(self, x):
    ...         x = self.conv1(x)
    ...         x = self.avg_pool(x)
    ...         x = self.fc(x)
    ...         return x
    ...
    >>> model = AvgPool1dModel()
    >>> input_data = Tensor([[[1.0, 2.0, 3.0, 4.0]]], requires_grad=True)  # Shape: (1, 1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[[...] ]], grad=None)  # Output tensor after passing through the model

    # Backpropagation
    >>> output.backward(Tensor([[[1.0]]]))
    >>> print(input_data.grad)
    Tensor([[[0.5, 0.5, 0.0, 0.0]]])  # Gradients with respect to input_data
