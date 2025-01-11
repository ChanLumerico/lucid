DenseNet
========

.. toctree::
    :maxdepth: 1
    :hidden:

    densenet_121.rst
    densenet_169.rst
    densenet_201.rst
    densenet_264.rst

.. autoclass:: lucid.models.DenseNet

The `DenseNet` class implements the Dense Convolutional Network (DenseNet),
featuring dense connections, bottleneck layers, and transition layers.

This class serves as the base for various DenseNet variants such as DenseNet-121, 
DenseNet-169, etc.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.DenseNet(
        block_config: tuple[int],
        growth_rate: int = 32,
        num_init_features: int = 64,
        num_classes: int = 1000,
    )

Parameters
----------
- **block_config** (*tuple[int]*): 
  Specifies the number of layers in each dense block.

- **growth_rate** (*int*, optional):
  Number of output channels added by each dense layer. Default is 32.

- **num_init_features** (*int*, optional):
  Number of output channels from the initial convolution layer. Default is 64.

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.

Examples
--------

**Defining a DenseNet-121 model:**

.. code-block:: python

    from lucid.models import DenseNet

    model = DenseNet(
        block_config=(6, 12, 24, 16),  # DenseNet-121
        growth_rate=32,
        num_init_features=64,
        num_classes=1000
    )

    input_tensor = lucid.random.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)

**Using a custom configuration:**

.. code-block:: python

    model = DenseNet(
        block_config=(4, 8, 16, 12),
        growth_rate=24,
        num_init_features=48,
        num_classes=10
    )

.. note::

  DenseNet is a memory-intensive architecture due to dense connections. 
  Consider optimizing for memory usage in large-scale applications.
