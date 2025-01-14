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
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):  
  The input tensor of shape (N, C_in, H, W), where N is the batch size,  
  C_in is the input channels, H is the height, and W is the width.

- **weight** (*Tensor*):  
  The weight tensor of shape (C_out, C_in // groups, K_H, K_W),  
  where K_H and K_W are the kernel height and width. If `groups` > 1,  
  the number of input channels (`C_in`) is divided by `groups`, and each  
  group processes `C_in // groups` input channels independently,  
  with each group having its own set of filters.

- **bias** (*Tensor | None*, optional):  
  The bias tensor of shape (C_out,). If None, no bias is added. Default: None.

- **stride** (*int | tuple[int, ...]*, optional):  
  The stride of the convolution. Can be an integer or a tuple. Default: 1.

- **padding** (*int | tuple[int, ...]*, optional):  
  The amount of zero-padding added to all sides of the input. Default: 0.

- **dilation** (*int | tuple[int, ...]*, optional):  
  The spacing between kernel elements. A dilation value of 1 means no spacing,  
  and larger values increase the effective size of the kernel by spacing out its elements. Default: 1.

- **groups** (*int*, optional):  
  The number of groups to divide the input channels into. When `groups > 1`,  
  the convolution operates separately on each group of channels. If `C_in`  
  is divisible by `groups`, each group processes `C_in // groups` input channels independently,  
  and the output channels are concatenated. Depthwise convolution can be achieved  
  by setting `groups = C_in`. Default: 1.

Returns
-------

- **Tensor**:  
  The result of the 2D convolution operation,  
  with shape (N, C_out, H_out, W_out), where:

  .. math::

      H_{out} = \\frac{H + 2 \\cdot \\text{padding}[0] - 
      \\text{dilation}[0] \\cdot (K_H - 1) - 1}{\\text{stride}[0]} + 1

      W_{out} = \\frac{W + 2 \\cdot \\text{padding}[1] - 
      \\text{dilation}[1] \\cdot (K_W - 1) - 1}{\\text{stride}[1]} + 1

Examples
--------

**Basic Example**

Performing a simple 2D convolution:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    >>> weight = Tensor([[[[1.0, 0.5], [0.5, 1.0]]]])  # Shape: (1, 1, 2, 2)
    >>> bias = Tensor([0.0])  # Shape: (1,)
    >>> out = F.conv2d(input_, weight, bias, stride=1, padding=0, dilation=1, groups=1)
    >>> print(out)
    Tensor([[[[10.0]]]])

**Advanced Example with Dilation**

Using `conv2d` with a dilation factor:

.. code-block:: python

    >>> input_ = Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])  # Shape: (1, 1, 3, 3)
    >>> weight = Tensor([[[[1.0, 0.5], [0.5, 1.0]]]])  # Shape: (1, 1, 2, 2)
    >>> bias = Tensor([0.0])  # Shape: (1,)
    >>> out = F.conv2d(input_, weight, bias, stride=1, padding=0, dilation=2, groups=1)
    >>> print(out)
    Tensor([[[[6.0]]]])  # Shape: (1, 1, 1, 1)

In this example, the effective kernel size increases due to the dilation factor,  
spacing out the kernel elements and covering a wider range in the input tensor.

**Advanced Example with Groups**

Using `conv2d` with a group of 2:

.. code-block:: python

    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])  # Shape: (1, 2, 2, 2)
    >>> weight = Tensor([[[[1.0, 0.5], [0.5, 1.0]]], [[[0.5, 1.0], [1.0, 0.5]]]])  # Shape: (2, 1, 2, 2)
    >>> bias = Tensor([0.0, 0.0])  # Shape: (2,)
    >>> out = F.conv2d(input_, weight, bias, stride=1, padding=0, dilation=1, groups=2)
    >>> print(out)
    Tensor([[[[10.0]]], [[[19.0]]]])

In this example, the input tensor has two input channels, but since `groups=2`,  
each input channel is convolved independently using its corresponding filter.  
This is often used in depthwise convolutions in neural networks.
