nn.functional.avg_pool2d
========================

.. autofunction:: lucid.nn.functional.avg_pool2d

The `avg_pool2d` function performs a two-dimensional average pooling operation on the input tensor.

Function Signature
------------------

.. code-block:: python

    def avg_pool2d(
        input_: Tensor,
        kernel_size: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape (N, C, H, W), where N is the batch size, C is the channels, H is the height, and W is the width.
    
- **kernel_size** (*int | tuple[int, int]*, optional): 
    The size of the pooling window. Default: 1.
    
- **stride** (*int | tuple[int, int]*, optional): 
    The stride of the pooling operation. Default: 1.
    
- **padding** (*int | tuple[int, int]*, optional): 
    The amount of zero-padding added to all sides of the input. Default: 0.

Returns
-------

- **Tensor**: 
    The result of the 2D average pooling operation, with shape (N, C, H_out, W_out), where:

    .. math::

        H_{out} = \frac{H + 2 \cdot \text{padding}[0] - \text{kernel height}}{\text{stride}[0]} + 1
        
        W_{out} = \frac{W + 2 \cdot \text{padding}[1] - \text{kernel width}}{\text{stride}[1]} + 1

Examples
--------

Performing a simple 2D average pooling:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    >>> out = F.avg_pool2d(input_, kernel_size=2, stride=1, padding=0)
    >>> print(out)
    Tensor([[[[2.5]]]])
