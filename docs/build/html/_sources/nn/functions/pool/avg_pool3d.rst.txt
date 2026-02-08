nn.functional.avg_pool3d
========================

.. autofunction:: lucid.nn.functional.avg_pool3d

The `avg_pool3d` function performs a three-dimensional average pooling operation on the input tensor.

Function Signature
------------------

.. code-block:: python

    def avg_pool3d(
        input_: Tensor,
        kernel_size: int | tuple[int, int, int] = 1,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape (N, C, D, H, W), where N is the batch size, C is the channels, D is the depth, H is the height, and W is the width.
    
- **kernel_size** (*int | tuple[int, int, int]*, optional): 
    The size of the pooling window. Default: 1.
    
- **stride** (*int | tuple[int, int, int]*, optional): 
    The stride of the pooling operation. Default: 1.
    
- **padding** (*int | tuple[int, int, int]*, optional): 
    The amount of zero-padding added to all sides of the input. Default: 0.

Returns
-------

- **Tensor**: 
    The result of the 3D average pooling operation, with shape (N, C, D_out, H_out, W_out), where:

    .. math::

        D_{out} = \frac{D + 2 \cdot \text{padding}[0] - \text{kernel depth}}{\text{stride}[0]} + 1
        
        H_{out} = \frac{H + 2 \cdot \text{padding}[1] - \text{kernel height}}{\text{stride}[1]} + 1
        
        W_{out} = \frac{W + 2 \cdot \text{padding}[2] - \text{kernel width}}{\text{stride}[2]} + 1

Examples
--------

Performing a simple 3D average pooling:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])  # Shape: (1, 1, 1, 2, 2)
    >>> out = F.avg_pool3d(input_, kernel_size=1, stride=1, padding=0)
    >>> print(out)
    Tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])
