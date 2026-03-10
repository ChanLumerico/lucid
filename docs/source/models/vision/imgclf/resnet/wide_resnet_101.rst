wide_resnet_101
===============

.. autofunction:: lucid.models.wide_resnet_101

The `wide_resnet_101` function creates a Wide ResNet-101 architecture, a ResNet-101
variant with wider bottleneck layers for increased representational capacity.

**Total Parameters**: 126,886,696

Function Signature
------------------

.. code-block:: python

    def wide_resnet_101(num_classes: int = 1000, **kwargs) -> ResNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for the final fully connected layer. Default is 1000.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `ResNetConfig`, excluding the preset
  `block` and `layers` fields. Any provided `block_args` are merged with the
  default `{"base_width": 128}` wide bottleneck setting.

Returns
-------

- **ResNet**:
  A Wide ResNet-101 model configured with the specified number of output classes
  and additional options.

Architecture Details
--------------------

Wide ResNet-101 uses the preset `ResNetConfig(block="bottleneck", layers=[3, 4, 23, 3],
block_args={"base_width": 128})`.

.. note::

   The architecture comprises 4 stages with layers `[3, 4, 23, 3]` and wider bottleneck blocks.

Examples
--------

**Creating a Wide ResNet-101 model for ImageNet classification:**

.. code-block:: python

    import lucid.models as models

    model = models.wide_resnet_101(num_classes=1000)
    print(model)
