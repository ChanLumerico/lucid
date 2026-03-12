resnet_269
==========

.. autofunction:: lucid.models.resnet_269

Overview
--------

The `resnet_269` function constructs a ResNet-269 model, an ultra-deep residual network
built with pre-activation bottleneck blocks for large-scale image classification.

It uses the preset `ResNetConfig(block="preact_bottleneck", layers=[3, 30, 48, 8])`
and accepts additional `ResNetConfig` keyword overrides such as `in_channels`,
`stem_type`, `stem_width`, `avg_down`, `channels`, and `block_args`.

**Total Parameters**: 102,069,416

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnet_269(num_classes: int = 1000, **kwargs) -> ResNet:

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
  An instance of the ResNet-269 model.

Examples
--------

Creating a ResNet-269 model for 1000 classes:

.. code-block:: python

    model = resnet_269(num_classes=1000)
    print(model)

.. note::

  - `ResNet-269` uses `PreActBottleneck` with a stage configuration of `[3, 30, 48, 8]`.
  - The returned model is equivalent to `ResNet(ResNetConfig(...))` with the preset values above.
