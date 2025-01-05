wide_resnet_101
===============

.. autofunction:: lucid.models.wide_resnet_101

The `wide_resnet_101` function is a registered model that creates a Wide ResNet-101 
architecture. This is a variant of the ResNet-101 model with a wider bottleneck configuration, 
often used to enhance the representational capacity of the network.

**Total Parameters**: 126,886,696

Function Signature
------------------

.. code-block:: python

    def wide_resnet_101(num_classes: int = 1000, **kwargs) -> ResNet

Parameters
----------
- **num_classes** (*int*, optional):
  The number of output classes for the final fully connected layer. 
  Default is 1000, suitable for ImageNet classification.

- **kwargs** (*dict*, optional):
  Additional keyword arguments passed to the `ResNet` constructor. 
  These can include options for customization such as initialization or 
  model-specific settings.

Returns
-------
- **ResNet**:
  A Wide ResNet-101 model configured with the specified number of output classes 
  and additional options.

Architecture Details
---------------------
Wide ResNet-101 is characterized by its wider bottleneck layers compared 
to the standard ResNet-101. Specifically, the `base_width` parameter determines 
the width of the bottleneck layers, with a default value of 128 in this implementation.

.. note::

   The architecture comprises 4 stages with layers [3, 4, 23, 3], 
   corresponding to the number of residual blocks in each stage.

Examples
--------

**Creating a Wide ResNet-101 model for ImageNet classification:**

.. code-block:: python

    import lucid.models as models

    # Create the model
    model = models.wide_resnet_101(num_classes=1000)

    # Print model summary
    models.summarize(model)
