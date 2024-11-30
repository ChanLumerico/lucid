nn.functional.conv1d
====================

.. autofunction:: lucid.nn.functional.conv1d

The `conv1d` function performs a one-dimensional convolution operation on the input tensor. 
This is commonly used in applications like processing sequential data or time-series data.

Function Signature
------------------

.. code-block:: python

    def conv1d(
        input_: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape (N, C_in, L), where N is the batch size, 
    C_in is the input channels, and L is the length of the sequence.
    
- **weight** (*Tensor*): 
    The weight tensor of shape (C_out, C_in, K), where C_out is the output channels, 
    and K is the kernel size.
    
- **bias** (*Tensor*, optional): 
    The bias tensor of shape (C_out,). If None, no bias is added.

- **stride** (*int | tuple[int, ...]*, optional): 
    The stride of the convolution. Default: 1.
    
- **padding** (*int | tuple[int, ...]*, optional): 
    The amount of zero-padding added to both sides of the input. Default: 0.

Returns
-------

- **Tensor**: 
    The result of the 1D convolution operation, with shape (N, C_out, L_out), 
    where L_out is the output sequence length, computed as:

    .. math::
    
        L_{out} = \frac{L + 2 \cdot \text{padding} - 
        \text{kernel size}}{\text{stride}} + 1

Examples
--------

Performing a simple 1D convolution:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[1.0, 2.0, 3.0, 4.0]])  # Shape: (1, 1, 4)
    >>> weight = Tensor([[[1.0, 0.5]]])  # Shape: (1, 1, 2)
    >>> bias = Tensor([0.0])  # Shape: (1,)
    >>> out = F.conv1d(input_, weight, bias, stride=1, padding=0)
    >>> print(out)
    Tensor([[[2.0, 3.5, 5.0]]])
