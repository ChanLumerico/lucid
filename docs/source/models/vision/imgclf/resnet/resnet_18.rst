resnet_18
=========

.. autofunction:: lucid.models.resnet_18

Overview
--------

The `resnet_18` function constructs a ResNet-18 model, a lightweight residual network
suitable for image classification tasks.

It uses the preset `ResNetConfig(block="basic", layers=[2, 2, 2, 2])` and accepts
additional `ResNetConfig` keyword overrides such as `in_channels`, `stem_type`,
`stem_width`, `avg_down`, `channels`, and `block_args`.

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
  Additional keyword arguments forwarded to `ResNetConfig`, excluding the preset
  `block` and `layers` fields.

Returns
-------

- **ResNet**:
  An instance of the ResNet-18 model.

Examples
--------

Creating a ResNet-18 model for 1000 classes:

.. code-block:: python

    model = resnet_18(num_classes=1000)
    print(model)

.. note::

  - `ResNet-18` uses `BasicBlock` with a stage configuration of `[2, 2, 2, 2]`.
  - The returned model is equivalent to `ResNet(ResNetConfig(...))` with the preset values above.
