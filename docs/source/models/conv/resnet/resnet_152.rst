models.resnet_152
=================

.. autofunction:: lucid.models.resnet_152

Overview
--------

The `resnet_152` function constructs a ResNet-152 model, a very deep residual network 
suitable for complex and large-scale image classification tasks.

It uses `Bottleneck` as the building block and is designed for datasets with 
`num_classes` categories.

**Total Parameters**: 60,192,808

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnet_152(num_classes: int = 1000, **kwargs) -> ResNet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **ResNet**:
  An instance of the ResNet-152 model.

Examples
--------

Creating a ResNet-152 model for 1000 classes:

.. code-block:: python

    from resnet_functions import resnet_152

    model = resnet_152(num_classes=1000)
    print(model)

.. note::

  - `ResNet-152` uses a configuration of `[3, 8, 36, 3]` for its layers.
  - By default, it initializes weights internally unless specified otherwise through `kwargs`.
