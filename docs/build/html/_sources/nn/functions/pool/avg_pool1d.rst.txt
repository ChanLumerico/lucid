nn.functional.avg_pool1d
========================

.. autofunction:: lucid.nn.functional.avg_pool1d

The `avg_pool1d` function performs a one-dimensional average pooling operation on the input tensor.

Function Signature
------------------

.. code-block:: python

    def avg_pool1d(
        input_: Tensor,
        kernel_size: int | tuple[int] = 1,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape (N, C, L), where N is the batch size, C is the channels, and L is the length.
    
- **kernel_size** (*int | tuple[int]*, optional): 
    The size of the pooling window. Default: 1.
    
- **stride** (*int | tuple[int]*, optional): 
    The stride of the pooling operation. Default: 1.
    
- **padding** (*int | tuple[int]*, optional): 
    The amount of zero-padding added to both sides of the input. Default: 0.

Returns
-------

- **Tensor**: 
    The result of the 1D average pooling operation, with shape (N, C, L_out), where:

    .. math::

        L_{out} = \frac{L + 2 \cdot \text{padding} - \text{kernel size}}{\text{stride}} + 1

Examples
--------

Performing a simple 1D average pooling:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[1.0, 2.0, 3.0, 4.0]]])  # Shape: (1, 1, 4)
    >>> out = F.avg_pool1d(input_, kernel_size=2, stride=2, padding=0)
    >>> print(out)
    Tensor([[[1.5, 3.5]]])
