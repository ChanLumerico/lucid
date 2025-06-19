se_resnet_18
============

.. autofunction:: lucid.models.se_resnet_18

Overview
--------

The `se_resnet_18` function constructs an SE-ResNet-18 model, 
a lightweight residual network augmented with Squeeze-and-Excitation (SE) blocks for 
adaptive recalibration of feature maps. It is suitable for image classification tasks.

**Total Parameters**: 11,778,592

Function Signature
------------------

.. code-block:: python

    @register_model
    def se_resnet_18(num_classes: int = 1000, **kwargs) -> SENet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **SENet**:
  An instance of the SE-ResNet-18 model.

Examples
--------

Creating an SE-ResNet-18 model for 1000 classes:

.. code-block:: python

    model = se_resnet_18(num_classes=1000)
    print(model)

.. note::

  - `SE-ResNet-18` uses a configuration of `[2, 2, 2, 2]` for its layers.
  - Incorporates `_SEResNetModule` for SE operations.
  - Initializes weights internally unless specified otherwise through `kwargs`.
