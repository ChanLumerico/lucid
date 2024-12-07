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
        dilation: int | tuple[int, ...] = 1,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):  
  The input tensor of shape (N, C_in, L), where N is the batch size,  
  C_in is the input channels, and L is the length of the sequence.

- **weight** (*Tensor*):  
  The weight tensor of shape (C_out, C_in, K), where C_out is the output channels,  
  and K is the kernel size.

- **bias** (*Tensor | None*):  
  The bias tensor of shape (C_out,). If None, no bias is added. Default: None.

- **stride** (*int | tuple[int, ...]*):  
  The stride of the convolution. Can be an integer or a tuple. Default: 1.

- **padding** (*int | tuple[int, ...]*):  
  The amount of zero-padding added to both sides of the input. Default: 0.

- **dilation** (*int | tuple[int, ...]*):  
  The spacing between kernel elements. A dilation value of 1 means no spacing, and  
  larger values increase the effective size of the kernel by spacing out its elements. Default: 1.

Returns
-------

- **Tensor**:  
  The result of the 1D convolution operation, with shape (N, C_out, L_out),  
  where `L_out` is the output sequence length, computed as:

  .. math::

      L_{out} = \frac{L + 2 \cdot \text{padding} - 
      \text{dilation} \cdot (K - 1) - 1}{\text{stride}} + 1

Examples
--------

Performing a simple 1D convolution:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[1.0, 2.0, 3.0, 4.0]])  # Shape: (1, 1, 4)
    >>> weight = Tensor([[[1.0, 0.5]]])  # Shape: (1, 1, 2)
    >>> bias = Tensor([0.0])  # Shape: (1,)
    >>> out = F.conv1d(input_, weight, bias, stride=1, padding=0, dilation=1)
    >>> print(out)
    Tensor([[[2.0, 3.5, 5.0]]])

Advanced Example with Dilation
------------------------------

Using `conv1d` with a dilation factor:

.. code-block:: python

    >>> dilation = 2  # Dilation of 2
    >>> out = F.conv1d(input_, weight, bias, stride=1, padding=0, dilation=dilation)
    >>> print(out)
    Tensor([[[1.0, 4.0]]])

In this example, the effective kernel size increases due to the dilation factor, 
spacing out the kernel elements and covering a wider range in the input tensor.
