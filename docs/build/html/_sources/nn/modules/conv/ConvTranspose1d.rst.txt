nn.ConvTranspose1d
==================

.. autoclass:: lucid.nn.ConvTranspose1d

The `ConvTranspose1d` module performs the *transposed* version of the one-dimensional 
convolution operation, often referred to as *deconvolution*. It is commonly used in 
tasks requiring upsampling, such as audio generation, sequence modeling, and the 
decoder parts of autoencoders or GANs.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ConvTranspose1d(
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        output_padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True
    )

Parameters
----------
- **in_channels** (*int*): 
  Number of channels in the input tensor.

- **out_channels** (*int*): 
  Number of channels in the output tensor.

- **kernel_size** (*int* or *tuple[int, ...]*): 
  Size of the convolving kernel. Accepts an integer or tuple.

- **stride** (*int* or *tuple[int, ...]*, optional): 
  Stride of the convolution. Default is `1`.

- **padding** (*_PaddingStr*, *int*, or *tuple[int, ...]*, optional): 
  Zero-padding added to both sides of the input. Default is `0`.

- **output_padding** (*int* or *tuple[int, ...]*, optional): 
  Additional size added to the output shape. Default is `0`.

- **dilation** (*int* or *tuple[int, ...]*, optional): 
  Spacing between kernel elements. Default is `1`.

- **groups** (*int*, optional): 
  Number of blocked connections from input channels to output channels. Default is `1`.

- **bias** (*bool*, optional): 
  If `True`, adds a learnable bias. Default is `True`.

Attributes
----------
- **weight** (*Tensor*): 
  Learnable weights of shape `(in_channels, out_channels // groups, kernel_size)`.

- **bias** (*Tensor* or *None*): 
  Learnable bias of shape `(out_channels)`. None if `bias=False`.

Forward Calculation
-------------------
The `ConvTranspose1d` module calculates the *transposed* convolution using the 
specified parameters. This operation can be understood as a gradient of the standard 
`Conv1d` with respect to its input.

.. math::

    \mathbf{y} = \mathbf{x} \star \mathbf{W} + \mathbf{b}

Where:

- :math:`\mathbf{x}` is the input tensor of shape :math:`(N, C_{in}, L_{in})`.
- :math:`\mathbf{W}` is the weight tensor of shape :math:`(C_{in}, C_{out} / \text{groups}, K)`.
- :math:`\mathbf{b}` is the bias tensor of shape :math:`(C_{out})`, if applicable.
- :math:`\mathbf{y}` is the output tensor of shape :math:`(N, C_{out}, L_{out})`.

Backward Gradient Calculation
-----------------------------
For the transposed convolution, gradients are propagated as:

- **w.r.t.** :math:`\mathbf{x}`:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \text{conv1d}(\mathbf{y}, \mathbf{W}_{\text{rev}})

- **w.r.t.** :math:`\mathbf{W}`:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{W}} = \text{cross-correlation}(\mathbf{x}, \mathbf{y})

- **w.r.t.** :math:`\mathbf{b}`:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{b}} = \sum \mathbf{y},\quad \text{over spatial dimensions}

Examples
--------

**Basic usage of `ConvTranspose1d`:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> x = Tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)  # Shape: (1, 1, 3)
    >>> deconv1d = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, bias=False)
    >>> print(deconv1d.weight)
    Tensor([[[1.0, 0.5]]], requires_grad=True)  # Example weights
    >>> y = deconv1d(x)
    >>> print(y)
    Tensor([[[1.0, 2.5, 3.5, 1.5]]])  # Shape: (1, 1, 4)

    # Backward pass
    >>> y.backward()
    >>> print(x.grad)
    Tensor([[[1.0, 0.5, 0.0]]])
    >>> print(deconv1d.weight.grad)
    Tensor([[[1.0, 2.0, 3.0]]])

**Stacked usage in a decoder:**

.. code-block:: python

    >>> class Decoder(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.deconv1 = nn.ConvTranspose1d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1)
    ...         self.relu = nn.ReLU()
    ...
    ...     def forward(self, x):
    ...         x = self.deconv1(x)
    ...         x = self.relu(x)
    ...         return x
    >>>
    >>> model = Decoder()
    >>> z = Tensor.randn(1, 4, 5, requires_grad=True)
    >>> output = model(z)
    >>> output.backward()
