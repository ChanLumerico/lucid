SENet
======

.. toctree::
    :maxdepth: 1
    :hidden:

    SENetConfig.rst
    se_resnet_18.rst
    se_resnet_34.rst
    se_resnet_50.rst
    se_resnet_101.rst
    se_resnet_152.rst
    se_resnext_50_32x4d.rst
    se_resnext_101_32x4d.rst
    se_resnext_101_32x8d.rst
    se_resnext_101_64x4d.rst

|convnet-badge|

.. autoclass:: lucid.models.SENet

The `SENet` class augments residual blocks with squeeze-and-excitation modules.
`SENetConfig` supports both the SE-ResNet basic-block family and the bottleneck-based
SE-ResNet / SE-ResNeXt variants through a single configuration surface.

Class Signature
---------------

.. code-block:: python

    class SENet(ResNet):
        def __init__(self, config: SENetConfig)

Parameters
----------

- **config** (*SENetConfig*):
  Configuration object describing the residual block family, stage depths, squeeze-and-excitation
  reduction ratio, grouped bottleneck settings, and shared ResNet backbone options.
