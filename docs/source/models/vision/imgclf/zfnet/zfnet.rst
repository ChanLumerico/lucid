zfnet
=====

.. autofunction:: lucid.models.zfnet

The `zfnet` function in `lucid.models` provides a convenient way to 
create an instance of the `ZFNet` module, a convolutional neural network 
designed for image classification with enhanced feature learning. It builds
`ZFNetConfig` internally and forwards extra keyword arguments to that config.

**Total Parameters**: 62,357,608

Function Signature
------------------

.. code-block:: python

    @register_model
    def zfnet(num_classes: int = 1000, **kwargs) -> ZFNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `ZFNetConfig`, such as
  `in_channels`, `dropout`, or `classifier_hidden_features`.

Returns
-------

- **ZFNet**:
  An instance of the `ZFNet` module configured with the specified number 
  of classes and any additional config overrides.

Examples
--------

**Creating a Default ZFNet Model**

.. code-block:: python

    import lucid.models as models

    # Create a ZFNet model with 1000 output classes
    model = models.zfnet()

    print(model)  # Displays the ZFNet architecture

**Custom Number of Classes**

.. code-block:: python

    # Create a ZFNet model with 10 output classes
    model = models.zfnet(num_classes=10)

    print(model)  # Displays the ZFNet architecture with modified output

**Custom Config Overrides**

.. code-block:: python

    model = models.zfnet(
        num_classes=10,
        in_channels=1,
        dropout=0.25,
        classifier_hidden_features=(512, 256),
    )

    print(model.config)
