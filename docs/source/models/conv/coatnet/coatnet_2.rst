coatnet_2
=========

.. autofunction:: lucid.models.coatnet_2

The `coatnet_2` function registers and returns an instance of the 
CoAtNet model with a predefined configuration. It follows the CoAtNet-2 
variant architecture, which balances convolutional and transformer 
layers for efficient image classification.

**Total Parameters**: 82,516,096

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_2(num_classes: int = 1000, **kwargs) -> CoAtNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional arguments passed to the `CoAtNet` constructor.

Configuration
-------------

The `coatnet_2` model follows the configuration from the original CoAtNet paper. 
The number of blocks and channels in each stage are summarized in the table below:

+-------------+-------------+------------+------------+
| Stage       | Block Type  | Blocks     | Channels   |
+=============+=============+============+============+
| Stage 1     | Convolution | 2          | 128        |
+-------------+-------------+------------+------------+
| Stage 2     | Convolution | 6          | 256        |
+-------------+-------------+------------+------------+
| Stage 3     | Transformer | 14         | 512        |
+-------------+-------------+------------+------------+
| Stage 4     | Transformer | 2          | 1024       |
+-------------+-------------+------------+------------+

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Instantiate coatnet_2 with default 1000 classes
    model = models.coatnet_2(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)
    print(output.shape)  # Shape: (1, 1000)

**Custom Number of Classes**

.. code-block:: python

    # Instantiate coatnet_2 for 10-class classification
    model = models.coatnet_2(num_classes=10)
    input_ = lucid.random.randn(1, 3, 224, 224)
    output = model(input_)
    print(output.shape)  # Shape: (1, 10)

