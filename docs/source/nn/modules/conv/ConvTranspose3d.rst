nn.ConvTranspose3d
==================

.. autoclass:: lucid.nn.ConvTranspose3d

The `ConvTranspose3d` module applies a transposed 3D convolution over volumetric inputs, 
such as videos or 3D medical images. It is especially useful in 3D autoencoders or 
volumetric segmentation networks.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ConvTranspose3d(
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: _PaddingStr | int | tuple[int, int, int] = 0,
        output_padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True
    )

Parameters
----------
- **in_channels** (*int*): 
  Number of channels in the input image.

- **out_channels** (*int*): 
  Number of channels in the output image.

- **kernel_size** (*int* or *tuple[int, int, int]*): 
  Size of the convolving kernel.

- **stride** (*int* or *tuple[int, int, int]*, optional): 
  Stride of the convolution. Default is `1`.

- **padding** (*_PaddingStr*, *int*, or *tuple[int, int, int]*, optional): 
  Zero-padding added to both sides of the input. Default is `0`.

- **output_padding** (*int* or *tuple[int, int, int]*, optional): 
  Additional size added to the output. Default is `0`.

- **dilation** (*int* or *tuple[int, int, int]*, optional): 
  Spacing between kernel elements. Default is `1`.

- **groups** (*int*, optional): 
  Number of blocked connections. Default is `1`.

- **bias** (*bool*, optional): 
  If `True`, adds a learnable bias. Default is `True`.

Attributes
----------
- **weight** (*Tensor*): 
    Shape: `(in_channels, out_channels // groups, D, H, W)`

- **bias** (*Tensor* or *None*): 
    Shape: `(out_channels,)`, if enabled.

Forward Calculation
-------------------
.. math::

    \mathbf{y} = \mathbf{x} \star \mathbf{W} + \mathbf{b}

- :math:`\mathbf{x}`: input shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
- :math:`\mathbf{W}`: weight shape :math:`(C_{in}, C_{out}/\text{groups}, kD, kH, kW)`
- :math:`\mathbf{y}`: output shape :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

Backward Gradient Calculation
-----------------------------
Follows same rules as `ConvTranspose2d`, extended to 3D.

Examples
--------

**Basic 3D upsampling:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> x = Tensor.randn(1, 4, 8, 8, 8, requires_grad=True)
    >>> deconv3d = nn.ConvTranspose3d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1)
    >>> y = deconv3d(x)
    >>> print(y.shape)
    (1, 2, 16, 16, 16)

    # Backprop
    >>> y.backward()
    >>> print(x.grad.shape)
    (1, 4, 8, 8, 8)

**Within a full 3D decoder model:**

.. code-block:: python

    >>> class Decoder3D(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.deconv = nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1)
    ...         self.relu = nn.ReLU()
    ...
    ...     def forward(self, x):
    ...         return self.relu(self.deconv(x))

    >>> model = Decoder3D()
    >>> z = Tensor.randn(2, 8, 4, 4, 4, requires_grad=True)
    >>> out = model(z)
    >>> out.backward()
