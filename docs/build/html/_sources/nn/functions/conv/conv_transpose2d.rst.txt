nn.functional.conv_transpose2d
==============================

.. autofunction:: lucid.nn.functional.conv_transpose2d

The `conv_transpose2d` function performs a two-dimensional transposed convolution  
(often called a deconvolution or fractionally strided convolution) on the input tensor.  
It is commonly used for upsampling feature maps in image-to-image tasks such as 
segmentation or generation.

Function Signature
------------------

.. code-block:: python

    def conv_transpose2d(
        input_: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):  
  The input tensor of shape (N, C_in, H, W), where N is the batch size,  
  C_in is the number of input channels, and H, W are height and width.

- **weight** (*Tensor*):  
  The filter tensor of shape (C_in, C_out // groups, K_H, K_W).  
  Each group applies its own set of filters to a split of the input.

- **bias** (*Tensor | None*):  
  Optional bias tensor of shape (C_out,). If None, no bias is added. Default: None.

- **stride** (*int | tuple[int, int]*):  
  The stride of the transposed convolution. Default: 1.

- **padding** (*int | tuple[int, int]*):  
  Zero-padding added to both sides of each dimension of the input. Default: 0.

- **dilation** (*int | tuple[int, int]*):  
  Spacing between kernel elements. Default: 1.

- **groups** (*int*):  
  Number of groups to divide the input channels into.  
  Depthwise transposed convolution is performed when `groups = C_in`. Default: 1.

Returns
-------

- **Tensor**:  
  The output tensor of shape (N, C_out, H_out, W_out), where:

  .. math::

      H_{out} = \text{stride}_H \cdot (H - 1) + 
      \text{dilation}_H \cdot (K_H - 1) - 2 \cdot \text{padding}_H + 1

  .. math::

      W_{out} = \text{stride}_W \cdot (W - 1) + 
      \text{dilation}_W \cdot (K_W - 1) - 2 \cdot \text{padding}_W + 1

Examples
--------

**Basic Example**

Performing a 2D transposed convolution:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    >>> weight = Tensor([[[[1.0, 0.5], [0.25, -0.25]]]])  # Shape: (1, 1, 2, 2)
    >>> bias = Tensor([0.0])
    >>> out = F.conv_transpose2d(input_, weight, bias, stride=1, padding=0, dilation=1, groups=1)
    >>> print(out)
    Tensor([[[[1.0, 2.5, 1.0],
              [3.75, 8.5, 3.5],
              [0.75, 1.5, -1.0]]]])

**Example with Stride 2**

Upsampling with stride:

.. code-block:: python

    >>> out = F.conv_transpose2d(input_, weight, bias, stride=2, padding=0)
    >>> print(out)
    Tensor([[[[1.0, 0.0, 2.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [3.0, 0.0, 4.0, 0.0],
              [0.0, 0.0, 0.0, 0.0]]]])

**Grouped Transposed Convolution**

.. code-block:: python

    >>> input_ = Tensor([[
    ...     [[1.0, 2.0], [3.0, 4.0]],  # C_in=2
    ...     [[-1.0, -2.0], [-3.0, -4.0]]
    ... ]])
    >>> weight = Tensor([
    ...     [[[1.0, 0.0], [0.0, 1.0]]],
    ...     [[[0.5, 0.5], [0.5, 0.5]]]
    ... ])  # Shape: (2, 1, 2, 2)
    >>> bias = Tensor([0.0, 0.0])
    >>> out = F.conv_transpose2d(input_, weight, bias, stride=1, padding=0, groups=2)
    >>> print(out.shape)
    (1, 2, 3, 3)

In this case, each input channel is convolved separately with its own filter due to `groups=2`.

.. note::

    `conv_transpose2d` is widely used in generative models such as GANs and decoder networks  
    in segmentation models where resolution needs to be increased progressively.
