models.resnet_101
=================

.. autofunction:: lucid.models.resnet_101

Overview
--------

The `resnet_101` function constructs a ResNet-101 model, a deep residual network 
suitable for large-scale image classification tasks.

It uses `Bottleneck` as the building block and is designed for datasets with 
`num_classes` categories.

**Total Parameters**: 44,549,160

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnet_101(num_classes: int = 1000, **kwargs) -> ResNet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **ResNet**:
  An instance of the ResNet-101 model.

Examples
--------

Creating a ResNet-101 model for 1000 classes:

.. code-block:: python

    from resnet_functions import resnet_101

    model = resnet_101(num_classes=1000)
    print(model)

.. note::

  - `ResNet-101` uses a configuration of `[3, 4, 23, 3]` for its layers.
  - By default, it initializes weights internally unless specified otherwise through `kwargs`.
