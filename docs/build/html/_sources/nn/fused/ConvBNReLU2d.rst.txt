nn.ConvBNReLU2d
===============

.. autoclass:: lucid.nn.ConvBNReLU2d

The `ConvBNReLU2d` module combines a 2D convolutional layer, 
batch normalization, and ReLU activation in one sequential block. 
It is particularly useful for processing 2D data such as images or spatial features.

Class Signature
---------------

.. code-block:: python

    class ConvBNReLU2d(
        in_channels: int,
        out_channels: int,
        num_features: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: _PaddingStr | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        conv_bias: bool = True,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        bn_affine: bool = True,
        track_running_stats: bool = True,
    ) -> None

Parameters
----------

- **in_channels** (*int*):
  Number of input channels.

- **out_channels** (*int*):
  Number of output channels (filters).

- **num_features** (*int*):
  Number of features for the batch normalization layer.

- **kernel_size** (*int | tuple[int, int]*):
  Size of the convolutional kernel.

- **stride** (*int | tuple[int, int]*, optional):
  Stride of the convolution. Default is 1.

- **padding** (*_PaddingStr | int | tuple[int, int]*, optional):
  Padding added to all sides of the input. Default is 0.

- **dilation** (*int | tuple[int, int]*, optional):
  Spacing between kernel elements. Default is 1.

- **groups** (*int*, optional):
  Number of groups for grouped convolution. Default is 1.

- **conv_bias** (*bool*, optional):
  If True, includes a bias term in the convolution layer. Default is True.

- **eps** (*float*, optional):
  Value added to the denominator for numerical stability in batch normalization. 
  Default is 1e-5.

- **momentum** (*float | None*, optional):
  Value used for the running mean and variance computation in batch normalization. 
  Default is 0.1.

- **bn_affine** (*bool*, optional):
  If True, batch normalization has learnable affine parameters. Default is True.

- **track_running_stats** (*bool*, optional):
  If True, tracks the running mean and variance. Default is True.

Attributes
----------

- **conv** (*nn.Conv2d*):
    The convolutional layer.

- **bn** (*nn.BatchNorm2d*):
    The batch normalization layer.

- **relu** (*nn.ReLU*):
    The ReLU activation function.

Forward Calculation
-------------------

The forward pass applies the following operations sequentially:

1. Convolution: Applies the convolution operation.
2. Batch Normalization: Normalizes the output of the convolution.
3. ReLU Activation: Applies the ReLU function element-wise.

Mathematically:

.. math::

    \text{output} = \text{ReLU}(\text{BatchNorm}(\text{Conv2D}(\text{input})))

Examples
--------

**Basic Example**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> layer = nn.ConvBNReLU2d(
    ...     in_channels=16, 
    ...     out_channels=32, 
    ...     num_features=32, 
    ...     kernel_size=(3, 3), 
    ...     stride=1, 
    ...     padding=1
    ... )
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (N, C_in, H, W)
    >>> output = layer(input_)
    >>> print(output)

**Custom Configuration**

.. code-block:: python

    >>> layer = nn.ConvBNReLU2d(
    ...     in_channels=8,
    ...     out_channels=16,
    ...     num_features=16,
    ...     kernel_size=(5, 5),
    ...     stride=2,
    ...     padding=2,
    ...     dilation=2
    ... )
    >>> print(layer)

.. note::

    - The layer combines three operations in a single class, simplifying sequential model definitions.
    - Ensure that the `num_features` in batch normalization matches `out_channels` 
      from the convolutional layer.
