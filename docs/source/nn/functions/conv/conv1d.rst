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
        groups: int = 1,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):  
  The input tensor of shape (N, C_in, L), where N is the batch size,  
  C_in is the input channels, and L is the length of the sequence.

- **weight** (*Tensor*):  
  The weight tensor of shape (C_out, C_in // groups, K), where C_out is the output channels,  
  and K is the kernel size. When `groups` > 1, the number of input channels (`C_in`)  
  is divided by `groups`, and each group has its own set of filters.

- **bias** (*Tensor | None*):  
  The bias tensor of shape (C_out,). If None, no bias is added. Default: None.

- **stride** (*int | tuple[int, ...]*):  
  The stride of the convolution. Can be an integer or a tuple. Default: 1.

- **padding** (*int | tuple[int, ...]*):  
  The amount of zero-padding added to both sides of the input. Default: 0.

- **dilation** (*int | tuple[int, ...]*):  
  The spacing between kernel elements. A dilation value of 1 means no spacing,  
  and larger values increase the effective size of the kernel by spacing out its elements. Default: 1.

- **groups** (*int*):  
  The number of groups to divide the input channels into. When `groups > 1`, the convolution operates  
  separately on each group of channels. If `C_in` is divisible by `groups`, each group processes  
  `C_in // groups` input channels independently, and the output channels are concatenated.  
  Depthwise convolution can be achieved by setting `groups = C_in`. Default: 1.

Returns
-------

- **Tensor**:  
  The result of the 1D convolution operation, with shape (N, C_out, L_out),  
  where `L_out` is the output sequence length, computed as:

  .. math::

      L_{out} = \\frac{L + 2 \\cdot \\text{padding} - 
      \\text{dilation} \\cdot (K - 1) - 1}{\\text{stride}} + 1

Examples
--------

**Basic Example**

Performing a simple 1D convolution:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[1.0, 2.0, 3.0, 4.0]])  # Shape: (1, 1, 4)
    >>> weight = Tensor([[[1.0, 0.5]]])  # Shape: (1, 1, 2)
    >>> bias = Tensor([0.0])  # Shape: (1,)
    >>> out = F.conv1d(input_, weight, bias, stride=1, padding=0, dilation=1, groups=1)
    >>> print(out)
    Tensor([[[2.0, 3.5, 5.0]]])

**Advanced Example with Dilation**

Using `conv1d` with a dilation factor:

.. code-block:: python

    >>> dilation = 2  # Dilation of 2
    >>> out = F.conv1d(input_, weight, bias, stride=1, padding=0, dilation=dilation, groups=1)
    >>> print(out)
    Tensor([[[1.0, 4.0]]])

In this example, the effective kernel size increases due to the dilation factor,  
spacing out the kernel elements and covering a wider range in the input tensor.

**Advanced Example with Groups**

Using `conv1d` with a group of 2:

.. code-block:: python

    >>> input_ = Tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])  # Shape: (1, 2, 4)
    >>> weight = Tensor([[[1.0, 0.5]], [[0.5, 1.0]]])  # Shape: (2, 1, 2) (Note: C_in / groups = 1)
    >>> bias = Tensor([0.0, 0.0])  # Shape: (2,)
    >>> out = F.conv1d(input_, weight, bias, stride=1, padding=0, dilation=1, groups=2)
    >>> print(out)
    Tensor([[[2.0, 3.5, 5.0], [5.5, 7.0, 8.5]]])

In this example, the input tensor has two input channels, but since `groups=2`,  
each input channel is convolved independently using its corresponding filter.  
This is often used in depthwise convolutions in neural networks.
