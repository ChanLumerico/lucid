resnet_50
=========

.. autofunction:: lucid.models.resnet_50

Overview
--------

The `resnet_50` function constructs a ResNet-50 model, a high-performance residual network 
suitable for a wide range of image classification tasks.

It uses `Bottleneck` as the building block and is designed for datasets with 
`num_classes` categories.

**Total Parameters**: 25,557,032

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnet_50(num_classes: int = 1000, **kwargs) -> ResNet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **ResNet**:
  An instance of the ResNet-50 model.

Examples
--------

Creating a ResNet-50 model for 1000 classes:

.. code-block:: python

    model = resnet_50(num_classes=1000)
    print(model)

.. note::

  - `ResNet-50` uses a configuration of `[3, 4, 6, 3]` for its layers.
  - By default, it initializes weights internally unless specified otherwise through `kwargs`.
