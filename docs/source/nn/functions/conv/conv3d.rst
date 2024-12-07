nn.functional.conv3d
====================

.. autofunction:: lucid.nn.functional.conv3d

The `conv3d` function performs a three-dimensional convolution operation on the input tensor.  
This is commonly used in 3D data processing, such as video processing or volumetric data.

Function Signature
------------------

.. code-block:: python

    def conv3d(
        input_: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):  
  The input tensor of shape (N, C_in, D, H, W), where N is the batch size,  
  C_in is the input channels, D is the depth, H is the height, and W is the width.

- **weight** (*Tensor*):  
  The weight tensor of shape (C_out, C_in, K_D, K_H, K_W),  
  where C_out is the output channels, and K_D, K_H, and K_W are the kernel depth, height, and width.

- **bias** (*Tensor | None*, optional):  
  The bias tensor of shape (C_out,). If None, no bias is added. Default: None.

- **stride** (*int | tuple[int, ...]*):  
  The stride of the convolution. Can be an integer or a tuple. Default: 1.

- **padding** (*int | tuple[int, ...]*):  
  The amount of zero-padding added to all sides of the input. Default: 0.

- **dilation** (*int | tuple[int, ...]*):  
  The spacing between kernel elements. A dilation value of 1 means no spacing,  
  and larger values increase the effective size of the kernel by spacing out its elements. Default: 1.

Returns
-------

- **Tensor**:  
  The result of the 3D convolution operation,  
  with shape (N, C_out, D_out, H_out, W_out), where:

  .. math::

      D_{out} = \frac{D + 2 \cdot \text{padding}[0] - \text{dilation}[0] 
      \cdot (K_D - 1) - 1}{\text{stride}[0]} + 1

      H_{out} = \frac{H + 2 \cdot \text{padding}[1] - \text{dilation}[1] 
      \cdot (K_H - 1) - 1}{\text{stride}[1]} + 1

      W_{out} = \frac{W + 2 \cdot \text{padding}[2] - \text{dilation}[2] 
      \cdot (K_W - 1) - 1}{\text{stride}[2]} + 1

Examples
--------

Performing a simple 3D convolution:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[[1.0]]]]])  # Shape: (1, 1, 1, 1, 1)
    >>> weight = Tensor([[[[[1.0]]]]])  # Shape: (1, 1, 1, 1, 1)
    >>> bias = Tensor([0.0])  # Shape: (1,)
    >>> out = F.conv3d(input_, weight, bias, stride=1, padding=0, dilation=1)
    >>> print(out)
    Tensor([[[[[1.0]]]]])

Advanced Example with Dilation
------------------------------

Using `conv3d` with a dilation factor:

.. code-block:: python

    >>> dilation = (2, 2, 2)  # Dilation in all dimensions
    >>> input_ = Tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])  # Shape: (1, 1, 2, 2, 2)
    >>> weight = Tensor([[[[[1.0, 0.5], [0.5, 1.0]]]]])  # Shape: (1, 1, 2, 2, 2)
    >>> bias = Tensor([0.0])  # Shape: (1,)
    >>> out = F.conv3d(input_, weight, bias, stride=1, padding=0, dilation=dilation)
    >>> print(out)
    Tensor([[[[[1.0]]]]])  # Output with dilation applied

In this example, the dilation factor increases the effective kernel size, 
allowing it to cover a larger region of the input tensor.
