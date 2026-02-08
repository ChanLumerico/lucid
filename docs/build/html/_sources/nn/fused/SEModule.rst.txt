nn.SEModule
===========

.. autoclass:: lucid.nn.SEModule

The `SEModule` (Squeeze-and-Excitation Module) applies channel-wise attention
by recalibrating feature maps with learned weights. It is typically used in
image processing tasks to improve representational power by adaptively
highlighting informative features.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.SEModule(in_channels: int, reduction: int = 16)

Parameters
----------
- **in_channels** (*int*):
    The number of input channels in the feature map.

- **reduction** (*int*, optional):
    Reduction ratio for the intermediate channel size in the bottleneck layer.
    Defaults to 16.

Attributes
----------
- **in_channels** (*int*):
    Stores the number of input channels.

- **reduction** (*int*):
    Stores the reduction ratio.

Forward Calculation
--------------------
The module performs the following operations:

1. Global Average Pooling (GAP) over the spatial dimensions of the input tensor.
   This reduces the input tensor shape from :math:`(N, C, *\text{spatial_dims})` 
   to :math:`(N, C, *1)`.

2. Two fully connected (FC) layers:

   - The first FC layer reduces the number of channels by the specified `reduction` ratio.
   - The second FC layer restores the number of channels to the original value.

   Activation functions:

   - ReLU after the first FC layer.
   - Sigmoid after the second FC layer to produce scaling factors.

3. Element-wise multiplication of the input tensor by the learned scaling factors.

.. math::

    \text{output} = \text{input} \cdot \sigma(f_2(\text{ReLU}(f_1(\text{GAP}(\text{input})))))

Where:

- :math:`f_1` and :math:`f_2` are the weights of the two FC layers.
- :math:`\sigma` is the sigmoid activation.

Backward Gradient Calculation
-----------------------------
Gradients are propagated through each of the layers and operations in the module.

Examples
--------

**Basic Example**:

.. code-block:: python

    >>> import lucid.nn as nn
    >>> module = nn.SEModule(in_channels=64, reduction=16)
    >>> input_tensor = Tensor(np.random.randn(8, 64, 32, 32))  # Shape: (N, C, H, W)
    >>> output = module(input_tensor)  # Forward pass
    >>> print(output.shape)
    (8, 64, 32, 32)

.. note::

  - The `SEModule` is commonly used in combination with convolutional layers in
    architectures such as ResNet and DenseNet.

  - The module introduces a slight computational overhead but significantly
    improves performance in tasks requiring fine-grained feature selection.

References
----------
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. *IEEE
  Transactions on Pattern Analysis and Machine Intelligence*, 42(8), 2011-2023.
