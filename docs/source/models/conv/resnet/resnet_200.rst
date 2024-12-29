models.resnet_200
=================

.. autofunction:: lucid.models.resnet_200

Overview
--------

The `resnet_200` function constructs a ResNet-200 model, a very deep residual network 
built with pre-activation bottleneck blocks, suitable for advanced image classification tasks.

It uses `PreActBottleneck` as the building block and is designed for datasets with 
`num_classes` categories.

**Total Parameters**: 64,669,864

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnet_200(num_classes: int = 1000, **kwargs) -> ResNet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **ResNet**:
  An instance of the ResNet-200 model.

Examples
--------

Creating a ResNet-200 model for 1000 classes:

.. code-block:: python

    from resnet_functions import resnet_200

    model = resnet_200(num_classes=1000)
    print(model)

.. note::

  - `ResNet-200` uses a configuration of `[3, 24, 36, 3]` for its layers.
  - By default, it initializes weights internally unless specified otherwise through `kwargs`.
