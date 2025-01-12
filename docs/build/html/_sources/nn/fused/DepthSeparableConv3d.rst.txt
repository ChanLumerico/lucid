nn.DepthSeparableConv3d
=======================

.. autoclass:: lucid.nn.DepthSeparableConv3d

The `DepthSeparableConv3d` module implements a 3D depthwise separable convolution layer, 
which is an efficient variation of the standard convolution. This layer performs a depthwise 
convolution followed by a pointwise convolution, reducing the number of parameters and computational cost.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.DepthSeparableConv3d(
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        base_act: Type[nn.Module] = nn.ReLU,
        do_act: bool = False,
        reversed: bool = False,
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

- **base_act** (*Type[nn.Module]*, optional):
  The activation module to use between the convolutions, 
  applied only if `do_act` is `True`. Default is `nn.ReLU`.

- **do_act** (*bool*, optional):
  If `True`, applies the `base_act` activation function between the 
  depthwise and pointwise convolutions. Default is `False`.

- **reversed** (*bool*, optional):
  If `True`, reverses the order of the convolutions, 
  performing pointwise convolution before depthwise convolution. Default is `False`.

- **bias** (*bool*, optional):
  If `True`, includes a bias term in the convolution. Default is `True`.

Attributes
----------
- **depthwise** (*nn.Conv3d*):
    The depthwise convolution layer.

- **pointwise** (*nn.Conv3d*):
    The pointwise convolution layer.

- **act** (*nn.Module*):
    The activation layer if `do_act` is `True`, otherwise `nn.Identity`.

Forward Calculation
--------------------
The `DepthSeparableConv3d` performs the following operations:

1. If `reversed` is `False`:
    .. math::
        \mathbf{y} = \text{pointwise}\big(\text{depthwise}(\mathbf{x})\big)

2. If `reversed` is `True`:
    .. math::
        \mathbf{y} = \text{depthwise}\big(\text{pointwise}(\mathbf{x})\big)

3. If `do_act` is `True`, applies the activation function between the convolutions.

Examples
--------

**Basic Usage**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = lucid.random.randn(1, 1, 4, 4, 4)
    >>> depthwise_separable = nn.DepthSeparableConv3d(
    ...     in_channels=1,
    ...     out_channels=2,
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1
    ... )
    >>> output = depthwise_separable(input_tensor)
    >>> print(output)
    Tensor([...], grad=None)

**Using Activation Between Convolutions**

.. code-block:: python

    >>> depthwise_separable = nn.DepthSeparableConv3d(
    ...     in_channels=1,
    ...     out_channels=2,
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1,
    ...     do_act=True
    ... )
    >>> output = depthwise_separable(input_tensor)
    >>> print(output)

**Reversed Convolution Order**

.. code-block:: python

    >>> depthwise_separable = nn.DepthSeparableConv3d(
    ...     in_channels=1,
    ...     out_channels=2,
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1,
    ...     reversed=True
    ... )
    >>> output = depthwise_separable(input_tensor)
    >>> print(output)

.. note::
        
    - Depthwise separable convolutions are more efficient than standard convolutions, 
      particularly for large inputs.
    - When using `reversed=True`, ensure that the input size is compatible with the 
      pointwise convolution first.
