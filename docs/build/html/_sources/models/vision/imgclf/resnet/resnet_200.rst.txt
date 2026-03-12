resnet_200
==========

.. autofunction:: lucid.models.resnet_200

Overview
--------

The `resnet_200` function constructs a ResNet-200 model, a very deep residual network
built with pre-activation bottleneck blocks for advanced image classification tasks.

It uses the preset `ResNetConfig(block="preact_bottleneck", layers=[3, 24, 36, 3])`
and accepts additional `ResNetConfig` keyword overrides such as `in_channels`,
`stem_type`, `stem_width`, `avg_down`, `channels`, and `block_args`.

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
  Additional keyword arguments forwarded to `ResNetConfig`, excluding the preset
  `block` and `layers` fields.

Returns
-------

- **ResNet**:
  An instance of the ResNet-200 model.

Examples
--------

Creating a ResNet-200 model for 1000 classes:

.. code-block:: python

    model = resnet_200(num_classes=1000)
    print(model)

.. note::

  - `ResNet-200` uses `PreActBottleneck` with a stage configuration of `[3, 24, 36, 3]`.
  - The returned model is equivalent to `ResNet(ResNetConfig(...))` with the preset values above.
