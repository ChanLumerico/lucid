nn.functional.conv_transpose3d
==============================

.. autofunction:: lucid.nn.functional.conv_transpose3d

The `conv_transpose3d` function performs a three-dimensional transposed convolution  
(commonly known as deconvolution or fractionally strided convolution) on a 5D input tensor.  
It is often used for volumetric data or 3D upsampling in applications like medical 
imaging and 3D reconstruction.

Function Signature
------------------

.. code-block:: python

    def conv_transpose3d(
        input_: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):  
  The input tensor of shape (N, C_in, D, H, W), where N is the batch size,  
  C_in is the number of input channels, and D, H, W are depth, height, and width.

- **weight** (*Tensor*):  
  The filter tensor of shape (C_in, C_out // groups, K_D, K_H, K_W).  
  Filters are grouped and applied to subsets of the input.

- **bias** (*Tensor | None*):  
  Optional bias tensor of shape (C_out,). Default: None.

- **stride** (*int | tuple[int, int, int]*):  
  Stride of the transposed convolution. Default: 1.

- **padding** (*int | tuple[int, int, int]*):  
  Zero-padding applied to all three spatial dimensions. Default: 0.

- **dilation** (*int | tuple[int, int, int]*):  
  Spacing between kernel elements in 3D. Default: 1.

- **groups** (*int*):  
  Number of groups for grouped transposed convolution. Default: 1.

Returns
-------

- **Tensor**:  
  The output tensor of shape (N, C_out, D_out, H_out, W_out), where:

  .. math::

      D_{out} = \text{stride}_D \cdot (D - 1) + 
      \text{dilation}_D \cdot (K_D - 1) - 2 \cdot \text{padding}_D + 1

  .. math::

      H_{out} = \text{stride}_H \cdot (H - 1) + 
      \text{dilation}_H \cdot (K_H - 1) - 2 \cdot \text{padding}_H + 1

  .. math::

      W_{out} = \text{stride}_W \cdot (W - 1) + 
      \text{dilation}_W \cdot (K_W - 1) - 2 \cdot \text{padding}_W + 1

Examples
--------

**Basic 3D Transposed Convolution**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[[1.0]]]]])  # Shape: (1, 1, 1, 1, 1)
    >>> weight = Tensor([[[[[1.0, 0.5], [0.25, -0.25]],
    ...                    [[0.1, -0.1], [0.0, 0.2]]]]])  # Shape: (1, 1, 2, 2, 2)
    >>> bias = Tensor([0.0])
    >>> out = F.conv_transpose3d(input_, weight, bias, stride=1, padding=0)
    >>> print(out.shape)
    (1, 1, 2, 2, 2)

**Upsampling with Stride > 1**

.. code-block:: python

    >>> out = F.conv_transpose3d(input_, weight, bias, stride=2, padding=0)
    >>> print(out.shape)
    (1, 1, 3, 3, 3)

**Grouped Transposed 3D Convolution**

.. code-block:: python

    >>> input_ = Tensor([[
    ...     [[[1.0]]], [[[2.0]]]
    ... ]])  # Shape: (1, 2, 1, 1, 1)
    >>> weight = Tensor([
    ...     [[[[[1.0]]]]],
    ...     [[[[[2.0]]]]]
    ... ])  # Shape: (2, 1, 1, 1, 1)
    >>> bias = Tensor([0.0, 0.0])
    >>> out = F.conv_transpose3d(input_, weight, bias, stride=1, padding=0, groups=2)
    >>> print(out.shape)
    (1, 2, 1, 1, 1)

.. warning::

    Ensure that `C_in` is divisible by `groups` and that the filter shape  
    matches the expected `(C_in, C_out // groups, K_D, K_H, K_W)` layout.

.. note::

    `conv_transpose3d` is especially useful in 3D reconstruction tasks  
    and in volumetric segmentation decoders.
