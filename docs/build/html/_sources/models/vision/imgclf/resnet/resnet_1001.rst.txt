resnet_1001
===========

.. autofunction:: lucid.models.resnet_1001

Overview
--------

The `resnet_1001` function constructs a ResNet-1001 model, an extremely deep residual network
built with pre-activation bottleneck blocks for research-scale image classification experiments.

It uses the preset `ResNetConfig(block="preact_bottleneck", layers=[3, 94, 94, 3])`
and accepts additional `ResNetConfig` keyword overrides such as `in_channels`,
`stem_type`, `stem_width`, `avg_down`, `channels`, and `block_args`.

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
  Additional keyword arguments forwarded to `ResNetConfig`, excluding the preset
  `block` and `layers` fields.

Returns
-------

- **ResNet**:
  An instance of the ResNet-1001 model.

Examples
--------

Creating a ResNet-1001 model for 1000 classes:

.. code-block:: python

    model = resnet_1001(num_classes=1000)
    print(model)

.. note::

  - `ResNet-1001` uses `PreActBottleneck` with a stage configuration of `[3, 94, 94, 3]`.
  - The returned model is equivalent to `ResNet(ResNetConfig(...))` with the preset values above.
