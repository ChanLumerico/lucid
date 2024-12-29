models.resnet_18
================

.. autofunction:: lucid.models.resnet_18

Overview
--------

The `resnet_18` function constructs a ResNet-18 model, a lightweight residual network 
suitable for image classification tasks.

It uses `BasicBlock` as the building block and is designed for datasets with 
`num_classes` categories.

**Total Parameters**: 11,689,512

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnet_18(num_classes: int = 1000, **kwargs) -> ResNet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **ResNet**:
  An instance of the ResNet-18 model.

Examples
--------

Creating a ResNet-18 model for 1000 classes:

.. code-block:: python

    from resnet_functions import resnet_18

    model = resnet_18(num_classes=1000)
    print(model)

.. note::

  - `ResNet-18` uses a configuration of `[2, 2, 2, 2]` for its layers.
  - By default, it initializes weights internally unless specified otherwise through `kwargs`.
