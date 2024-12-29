se_resnet_101
=============

.. autofunction:: lucid.models.se_resnet_101

Overview
--------

The `se_resnet_101` function constructs an SE-ResNet-101 model, 
a deep residual network with SE blocks for adaptive feature recalibration, 
designed for large-scale image classification tasks.

**Total Parameters**: 49,326,872

Function Signature
------------------

.. code-block:: python

    @register_model
    def se_resnet_101(num_classes: int = 1000, **kwargs) -> SENet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **SENet**:
  An instance of the SE-ResNet-101 model.

Examples
--------

Creating an SE-ResNet-101 model for 1000 classes:

.. code-block:: python

    model = se_resnet_101(num_classes=1000)
    print(model)

.. note::

  - `SE-ResNet-101` uses a configuration of `[3, 4, 23, 3]` for its layers.
  - Incorporates `_SEResNetBottleneck` for SE operations.
  - Initializes weights internally unless specified otherwise through `kwargs`.
