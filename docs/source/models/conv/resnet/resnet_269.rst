models.resnet_269
=================

.. autofunction:: lucid.models.resnet_269

Overview
--------

The `resnet_269` function constructs a ResNet-269 model, an ultra-deep residual network 
built with pre-activation bottleneck blocks, suitable for large-scale and complex image classification tasks.

It uses `PreActBottleneck` as the building block and is designed for datasets with 
`num_classes` categories.

**Total Parameters**: 102,069,416

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnet_269(num_classes: int = 1000, **kwargs) -> ResNet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **ResNet**:
  An instance of the ResNet-269 model.

Examples
--------

Creating a ResNet-269 model for 1000 classes:

.. code-block:: python

    from resnet_functions import resnet_269

    model = resnet_269(num_classes=1000)
    print(model)

.. note::

  - `ResNet-269` uses a configuration of `[3, 30, 48, 8]` for its layers.
  - By default, it initializes weights internally unless specified otherwise through `kwargs`.
