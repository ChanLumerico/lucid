models.resnet_1001
==================

.. autofunction:: lucid.models.resnet_1001

Overview
--------

The `resnet_1001` function constructs a ResNet-1001 model, an extremely deep residual network 
built with pre-activation bottleneck blocks, designed for large-scale and computationally 
intensive image classification tasks.

It uses `PreActBottleneck` as the building block and is designed for datasets with 
`num_classes` categories.

**Total Parameters**: 149,071,016

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnet_1001(num_classes: int = 1000, **kwargs) -> ResNet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **ResNet**:
  An instance of the ResNet-1001 model.

Examples
--------

Creating a ResNet-1001 model for 1000 classes:

.. code-block:: python

    from resnet_functions import resnet_1001

    model = resnet_1001(num_classes=1000)
    print(model)

.. note::

  - `ResNet-1001` uses a configuration of `[3, 94, 94, 3]` for its layers.
  - By default, it initializes weights internally unless specified otherwise through `kwargs`.
