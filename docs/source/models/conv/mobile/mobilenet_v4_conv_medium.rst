mobilenet_v4_conv_medium
========================

.. autofunction:: lucid.models.mobilenet_v4_conv_medium

Overview
--------
The `mobilenet_v4_conv_medium` function provides an instance of the `MobileNet_V4` model configured for a 
medium-scale convolutional variant. This version is designed to offer a balanced compromise between computational 
efficiency and performance, making it well-suited for devices with moderate resource constraints.

**Total Parameters**: 9,715,512

Parameters
----------
- **num_classes** (*int*, optional):  
  Specifies the number of output classes. The default is 1000, 
  typically used for ImageNet classification.

- **\*\*kwargs** (*dict*):  
  Additional keyword arguments that allow further customization of the model configuration. These can be used to 
  override default settings and fine-tune the architecture for specific application requirements.

Usage Example
-------------
.. code-block:: python

    >>> import lucid.nn as nn
    >>> model = nn.mobilenet_v4_conv_medium(num_classes=1000)
    >>> print(model)

Details
-------
The `mobilenet_v4_conv_medium` function instantiates a MobileNet-v4 model variant optimized for a medium-scale 
convolutional configuration. This variant is particularly suitable for scenarios where a balance 
between computational cost and model accuracy is required.

.. note::

   For scenarios requiring different performance or resource trade-offs, consider using the base `MobileNet_V4` 
   class with a custom configuration or exploring other registered MobileNet_V4 variants.