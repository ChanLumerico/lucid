nn.DepthSeparableConv1d
=======================

.. autoclass:: lucid.nn.DepthSeparableConv1d

The `DepthSeparableConv1d` module implements a 1D depthwise separable convolution layer, 
which is an efficient variation of the standard convolution. This layer performs a depthwise 
convolution followed by a pointwise convolution, reducing the number of parameters and computational cost.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.DepthSeparableConv1d(
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        bias: bool = True
    ) -> None

Parameters
----------
- **in_channels** (*int*):
  Number of input channels.

- **out_channels** (*int*):
  Number of output channels.

- **kernel_size** (*int | tuple[int, ...]*):
  Size of the convolution kernel.

- **stride** (*int | tuple[int, ...]*, optional):
  Stride of the convolution. Default is `1`.

- **padding** (*_PaddingStr | int | tuple[int, ...]*, optional):
  Amount of padding. Default is `0`.

- **dilation** (*int | tuple[int, ...]*, optional):
  Spacing between kernel elements. Default is `1`.

- **bias** (*bool*, optional):
  If `True`, includes a bias term in the convolution. Default is `True`.

Attributes
----------
- **depthwise** (*nn.Conv1d*):
    The depthwise convolution layer.

- **pointwise** (*nn.Conv1d*):
    The pointwise convolution layer.

Forward Calculation
--------------------
The `DepthSeparableConv1d` performs the following operations:

.. math::
      
    \mathbf{y} = \text{pointwise}\big(\text{depthwise}(\mathbf{x})\big)

Examples
--------

**Basic Usage**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = lucid.random.randn(1, 1, 4)
    >>> depthwise_separable = nn.DepthSeparableConv1d(
    ...     in_channels=1,
    ...     out_channels=2,
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1
    ... )
    >>> output = depthwise_separable(input_tensor)
    >>> print(output)
    Tensor([...], grad=None)

.. note::
        
    - Depthwise separable convolutions are more efficient than standard convolutions, 
      particularly for large inputs.
