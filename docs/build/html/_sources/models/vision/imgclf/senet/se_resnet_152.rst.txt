se_resnet_152
=============

.. autofunction:: lucid.models.se_resnet_152

Overview
--------

The `se_resnet_152` function constructs an SE-ResNet-152 model, 
a deep and accurate residual network with SE blocks, suitable for large-scale and 
complex image classification tasks.

**Total Parameters**: 66,821,848

Function Signature
------------------

.. code-block:: python

    @register_model
    def se_resnet_152(num_classes: int = 1000, **kwargs) -> SENet:

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000.

- **kwargs**:
  Additional keyword arguments to customize the model.

Returns
-------

- **SENet**:
  An instance of the SE-ResNet-152 model.

Examples
--------

Creating an SE-ResNet-152 model for 1000 classes:

.. code-block:: python

    model = se_resnet_152(num_classes=1000)
    print(model)

.. note::

  - `SE-ResNet-152` uses a configuration of `[3, 8, 36, 3]` for its layers.
  - Incorporates `_SEResNetBottleneck` for SE operations.
  - Initializes weights internally unless specified otherwise through `kwargs`.
