coatnet_3
=========

.. autofunction:: lucid.models.coatnet_3

The `coatnet_3` function registers and returns an instance of the 
CoAtNet model with a predefined configuration. It follows the CoAtNet-3 
variant architecture, which balances convolutional and transformer 
layers for efficient image classification.

**Total Parameters**: 157,790,656

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_3(num_classes: int = 1000, **kwargs) -> CoAtNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional arguments passed to the `CoAtNet` constructor.

Configuration
-------------

The `coatnet_3` model follows the configuration from the original CoAtNet paper. 
The number of blocks and channels in each stage are summarized in the table below:

+-------------+-------------+------------+------------+
| Stage       | Block Type  | Blocks     | Channels   |
+=============+=============+============+============+
| Stage 1     | Convolution | 2          | 192        |
+-------------+-------------+------------+------------+
| Stage 2     | Convolution | 6          | 384        |
+-------------+-------------+------------+------------+
| Stage 3     | Transformer | 14         | 768        |
+-------------+-------------+------------+------------+
| Stage 4     | Transformer | 2          | 1536       |
+-------------+-------------+------------+------------+

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Instantiate coatnet_3 with default 1000 classes
    model = models.coatnet_3(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)
    print(output.shape)  # Shape: (1, 1000)

**Custom Number of Classes**

.. code-block:: python

    # Instantiate coatnet_3 for 10-class classification
    model = models.coatnet_3(num_classes=10)
    input_ = lucid.random.randn(1, 3, 224, 224)
    output = model(input_)
    print(output.shape)  # Shape: (1, 10)

