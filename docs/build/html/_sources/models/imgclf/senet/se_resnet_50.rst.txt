se_resnet_50
============

.. autofunction:: lucid.models.se_resnet_50

Overview
--------

The `se_resnet_50` function constructs an SE-ResNet-50 model, 
a deep residual network with SE blocks, suitable for high-accuracy image classification tasks.

**Total Parameters**: 28,088,024

Function Signature
------------------

.. code-block:: python

    @register_model
    def se_resnet_50(num_classes: int = 1000, **kwargs) -> SENet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **SENet**:
  An instance of the SE-ResNet-50 model.

Examples
--------

Creating an SE-ResNet-50 model for 1000 classes:

.. code-block:: python

    model = se_resnet_50(num_classes=1000)
    print(model)

.. note::

  - `SE-ResNet-50` uses a configuration of `[3, 4, 6, 3]` for its layers.
  - Incorporates `_SEResNetBottleneck` for SE operations.
  - Initializes weights internally unless specified otherwise through `kwargs`.
