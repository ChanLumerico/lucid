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
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape (N, C_in, D, H, W), where D is the depth, 
    H is the height, and W is the width.
    
- **weight** (*Tensor*): 
    The weight tensor of shape (C_out, C_in, K_D, K_H, K_W), 
    where K_D, K_H, and K_W are the kernel depth, height, and width.
    
- **bias** (*Tensor*, optional): 
    The bias tensor of shape (C_out,). If None, no bias is added.

- **stride** (*int | tuple[int, ...]*, optional): 
    The stride of the convolution. Default: 1.
    
- **padding** (*int | tuple[int, ...]*, optional): 
    The amount of zero-padding added to all sides of the input. Default: 0.

Returns
-------

- **Tensor**: 
    The result of the 3D convolution operation, 
    with shape (N, C_out, D_out, H_out, W_out), where:

    .. math::

        D_{out} = \frac{D + 2 \cdot \text{padding}[0] - 
        \text{kernel depth}}{\text{stride}[0]} + 1

        H_{out} = \frac{H + 2 \cdot \text{padding}[1] - 
        \text{kernel height}}{\text{stride}[1]} + 1

        W_{out} = \frac{W + 2 \cdot \text{padding}[2] - 
        \text{kernel width}}{\text{stride}[2]} + 1

Examples
--------

Performing a simple 3D convolution:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[[1.0]]]]])  # Shape: (1, 1, 1, 1, 1)
    >>> weight = Tensor([[[[[1.0]]]]])  # Shape: (1, 1, 1, 1, 1)
    >>> bias = Tensor([0.0])  # Shape: (1,)
    >>> out = F.conv3d(input_, weight, bias, stride=1, padding=0)
    >>> print(out)
    Tensor([[[[[1.0]]]]])
