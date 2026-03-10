resnet_101
==========

.. autofunction:: lucid.models.resnet_101

Overview
--------

The `resnet_101` function constructs a ResNet-101 model, a deep residual network
suitable for large-scale image classification tasks.

It uses the preset `ResNetConfig(block="bottleneck", layers=[3, 4, 23, 3])` and accepts
additional `ResNetConfig` keyword overrides such as `in_channels`, `stem_type`,
`stem_width`, `avg_down`, `channels`, and `block_args`.

**Total Parameters**: 44,549,160

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnet_101(num_classes: int = 1000, **kwargs) -> ResNet:

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
  An instance of the ResNet-101 model.

Examples
--------

Creating a ResNet-101 model for 1000 classes:

.. code-block:: python

    model = resnet_101(num_classes=1000)
    print(model)

.. note::

  - `ResNet-101` uses `Bottleneck` with a stage configuration of `[3, 4, 23, 3]`.
  - The returned model is equivalent to `ResNet(ResNetConfig(...))` with the preset values above.
