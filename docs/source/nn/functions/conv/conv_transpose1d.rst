nn.functional.conv_transpose1d
==============================

.. autofunction:: lucid.nn.functional.conv_transpose1d

The `conv_transpose1d` function performs a one-dimensional transposed convolution 
(also known as fractionally strided convolution or deconvolution) on the input tensor. 
It is commonly used in upsampling operations such as in decoder parts of autoencoders 
or generative models.

Function Signature
------------------

.. code-block:: python

    def conv_transpose1d(
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
  C_in is the input channels, and L is the length of the input sequence.

- **weight** (*Tensor*):  
  The filter tensor of shape (C_in, C_out // groups, K), where K is the kernel size.  
  The number of input channels must match, and each group has `C_out // groups` filters.

- **bias** (*Tensor | None*):  
  Optional bias tensor of shape (C_out,). If None, no bias is added. Default: None.

- **stride** (*int | tuple[int, ...]*):  
  The stride of the transposed convolution. Default: 1.

- **padding** (*int | tuple[int, ...]*):  
  Zero-padding applied to both sides of the input. Default: 0.

- **dilation** (*int | tuple[int, ...]*):  
  Spacing between the kernel elements. Default: 1.

- **groups** (*int*):  
  Number of blocked connections from input channels to output channels.  
  For `groups > 1`, each group operates independently. Depthwise transposed convolution  
  is performed when `groups = C_in`. Default: 1.

Returns
-------

- **Tensor**:  
  The result of the transposed convolution, with shape (N, C_out, L_out),  
  where `L_out` is computed as:

  .. math::

      L_{out} = \text{stride} \cdot (L - 1) + 
      \text{dilation} \cdot (K - 1) - 2 \cdot \text{padding} + 1

Examples
--------

**Basic Example**

Performing a simple transposed convolution:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[1.0, 2.0, 3.0]]])  # Shape: (1, 1, 3)
    >>> weight = Tensor([[[1.0, 0.5]]])       # Shape: (1, 1, 2)
    >>> bias = Tensor([0.0])
    >>> out = F.conv_transpose1d(input_, weight, bias, stride=1, padding=0, dilation=1, groups=1)
    >>> print(out)
    Tensor([[[1.0, 2.5, 4.0, 1.5]]])

**Example with Stride > 1**

Upsampling by using stride:

.. code-block:: python

    >>> out = F.conv_transpose1d(input_, weight, bias, stride=2, padding=0, dilation=1, groups=1)
    >>> print(out)
    Tensor([[[1.0, 0.5, 2.0, 1.0, 3.0, 1.5]]])

In this case, transposed convolution introduces zeros between input elements due to stride.

**Grouped Transposed Convolution**

.. code-block:: python

    >>> input_ = Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # Shape: (1, 2, 3)
    >>> weight = Tensor([[[1.0, 0.5]], [[-1.0, 0.5]]])         # Shape: (2, 1, 2)
    >>> bias = Tensor([0.0, 0.0])
    >>> out = F.conv_transpose1d(input_, weight, bias, stride=1, padding=0, dilation=1, groups=2)
    >>> print(out)
    Tensor([[[1.0, 2.5, 4.0, 1.5], [-4.0, -3.5, -2.0, 3.0]]])

Here, each input channel is convolved with its own filter independently due to `groups=2`.

.. tip::

    Use `conv_transpose1d` when you need to increase the temporal resolution (upsample) 
    in 1D models, such as in decoder architectures or sequence generation tasks.
