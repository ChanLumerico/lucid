nn.functional.conv2d
====================

.. autofunction:: lucid.nn.functional.conv2d

The `conv2d` function performs a two-dimensional convolution operation on the input tensor. 
This is essential in image processing tasks and neural networks for computer vision.

Function Signature
------------------

.. code-block:: python

    def conv2d(
        input_: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape (N, C_in, H, W), where N is the batch size, 
    C_in is the input channels, H is the height, and W is the width.
    
- **weight** (*Tensor*): 
    The weight tensor of shape (C_out, C_in, K_H, K_W), 
    where K_H and K_W are the kernel height and width.
    
- **bias** (*Tensor*, optional): 
    The bias tensor of shape (C_out,). If None, no bias is added.

- **stride** (*int | tuple[int, ...]*, optional): 
    The stride of the convolution. Default: 1.
    
- **padding** (*int | tuple[int, ...]*, optional): 
    The amount of zero-padding added to all sides of the input. Default: 0.

Returns
-------

- **Tensor**: 
    The result of the 2D convolution operation, 
    with shape (N, C_out, H_out, W_out), where:

    .. math::

        H_{out} = \frac{H + 2 \cdot \text{padding}[0] - \text{kernel height}}{\text{stride}[0]} + 1
        W_{out} = \frac{W + 2 \cdot \text{padding}[1] - \text{kernel width}}{\text{stride}[1]} + 1

Examples
--------

Performing a simple 2D convolution:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    >>> weight = Tensor([[[[1.0, 0.5], [0.5, 1.0]]]])  # Shape: (1, 1, 2, 2)
    >>> bias = Tensor([0.0])  # Shape: (1,)
    >>> out = F.conv2d(input_, weight, bias, stride=1, padding=0)
    >>> print(out)
    Tensor([[[[10.0]]]])
