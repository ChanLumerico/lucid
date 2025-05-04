ResNet
======

.. toctree::
    :maxdepth: 1
    :hidden:

    resnet_18.rst
    resnet_34.rst
    resnet_50.rst
    resnet_101.rst
    resnet_152.rst
    resnet_200.rst
    resnet_269.rst
    resnet_1001.rst

    wide_resnet_50.rst
    wide_resnet_101.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.ResNet

The `ResNet` class provides an implementation of the ResNet architecture. It allows flexibility
in specifying custom block types, layer configurations, and hyperparameters, making it suitable
for a wide range of tasks in computer vision.

.. image:: resnet.png
    :width: 600
    :alt: ResNet architecture
    :align: center

Class Signature
---------------

.. code-block:: python

    class lucid.nn.ResNet(
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        stem_width: int = 64,
        stem_type: Literal["deep"] | None = None,
        channels: tuple[int] = (64, 128, 256, 512),
        block_args: dict[str, Any] = {},
    )

Parameters
----------
- **block** (*nn.Module*):
    The building block module used for the ResNet layers. Typically, 
    this is a residual block such as `BasicBlock` or `Bottleneck`.

- **layers** (*list[int]*):
    Specifies the number of blocks in each stage of the network.

- **num_classes** (*int*, optional):
    Number of output classes for the final fully connected layer. Default: 1000.

- **in_channels** (*int*, optional):
    Number of input channels for the input images. Default: 3.

- **stem_width** (*int*, optional):
    Number of output channels for the initial convolutional stem. Default: 64.

- **stem_type** (*Literal["deep"] | None*, optional):
    Specifies the type of stem. If "deep," a deeper stem with multiple layers is used.
    If None, a standard single-layer stem is used. Default: None.

- **channels** (*tuple[int]*, optional):
    Defines the output channel sizes for each stage of the network. 
    Default: (64, 128, 256, 512).

- **block_args** (*dict[str, Any]*, optional):
    Additional keyword arguments passed to the `block` module during construction.

Attributes
----------
- **stem** (*nn.Module*):
    The initial stem layer that processes the input tensor.

- **layers** (*list[nn.Module]*):
    A list of stages, each containing a sequence of blocks.

- **num_classes** (*int*):
    Stores the number of output classes.

- **block** (*nn.Module*):
    Stores the block type used for building the layers.

Forward Calculation
--------------------
The forward pass of the ResNet model includes:

1. **Stem**: Initial convolutional layers for feature extraction.
2. **Residual Stages**: Each stage consists of multiple blocks defined by the `layers` parameter.
3. **Global Pooling**: A global average pooling layer reduces the spatial dimensions.
4. **Classifier**: A fully connected layer maps the features to class scores.

.. math::

    \text{output} = \text{FC}(\text{GAP}(\text{ResidualBlocks}(\text{Stem}(\text{input}))))

Examples
--------

**Basic Example**:

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid.models.blocks import BasicBlock
    >>> layers = [3, 4, 6, 3]  # Configuration for ResNet-50
    >>> model = nn.ResNet(block=BasicBlock, layers=layers, num_classes=1000)
    >>> input_tensor = Tensor(np.random.randn(8, 3, 224, 224))  # Shape: (N, C, H, W)
    >>> output = model(input_tensor)  # Forward pass
    >>> print(output.shape)
    (8, 1000)

.. note::

   - The `ResNet` class supports flexible configurations for custom tasks by modifying the 
     `block`, `layers`, or `channels` parameters.
   - Adding a "deep" stem can improve feature extraction for larger or more complex datasets.
