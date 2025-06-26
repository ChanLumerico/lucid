nn.ConvTranspose2d
==================

.. autoclass:: lucid.nn.ConvTranspose2d

The `ConvTranspose2d` module performs the transposed 2D convolution 
(also called deconvolution) over a 2D input such as an image or feature map. 
This operation is commonly used for upsampling tasks in computer vision, 
such as semantic segmentation or image generation (e.g., GAN decoders).

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ConvTranspose2d(
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: _PaddingStr | int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True
    )

Parameters
----------
- **in_channels** (*int*): 
  Number of channels in the input image.

- **out_channels** (*int*): 
  Number of channels in the output image.

- **kernel_size** (*int* or *tuple[int, int]*): 
  Size of the convolving kernel.

- **stride** (*int* or *tuple[int, int]*, optional): 
  Stride of the convolution. Default is `1`.

- **padding** (*_PaddingStr*, *int*, or *tuple[int, int]*, optional): 
  Zero-padding added to both sides of the input. Default is `0`.

- **output_padding** (*int* or *tuple[int, int]*, optional): 
  Additional size added to the output. Default is `0`.

- **dilation** (*int* or *tuple[int, int]*, optional): 
  Spacing between kernel elements. Default is `1`.

- **groups** (*int*, optional): 
  Number of blocked connections. Default is `1`.

- **bias** (*bool*, optional): 
  If `True`, adds a learnable bias. Default is `True`.

Attributes
----------
- **weight** (*Tensor*): 
  Shape: `(in_channels, out_channels // groups, kH, kW)`.

- **bias** (*Tensor* or *None*): 
  Shape: `(out_channels,)`, if enabled.

Forward Calculation
-------------------
.. math::

    \mathbf{y} = \mathbf{x} \star \mathbf{W} + \mathbf{b}

- :math:`\mathbf{x}`: input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
- :math:`\mathbf{W}`: weight of shape :math:`(C_{in}, C_{out}/\text{groups}, kH, kW)`
- :math:`\mathbf{y}`: output of shape :math:`(N, C_{out}, H_{out}, W_{out})`

Backward Gradient Calculation
-----------------------------
- **w.r.t.** input:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \text{conv2d}(\mathbf{y}, \mathbf{W}_{\text{rev}})

- **w.r.t.** weight:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{W}} = \text{cross-correlation}(\mathbf{x}, \mathbf{y})

- **w.r.t.** bias:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{b}} = \sum \mathbf{y}

Examples
--------

**Basic use of `ConvTranspose2d`:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> x = Tensor.randn(1, 3, 16, 16, requires_grad=True)
    >>> deconv2d = nn.ConvTranspose2d(3, 2, kernel_size=3, stride=2, padding=1, output_padding=1)
    >>> y = deconv2d(x)
    >>> print(y.shape)
    (1, 2, 32, 32)

    # Backward pass
    >>> y.backward()
    >>> print(x.grad.shape)
    (1, 3, 16, 16)

**As part of decoder network:**

.. code-block:: python

    >>> class Decoder(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
    ...         self.relu = nn.ReLU()
    ...
    ...     def forward(self, x):
    ...         return self.relu(self.deconv1(x))
