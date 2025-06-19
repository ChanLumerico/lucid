mobilenet_v4_conv_small
=======================

.. autofunction:: lucid.models.mobilenet_v4_conv_small

Overview
--------
The `mobilenet_v4_conv_small` function returns an instance of the `MobileNet_V4` 
model tailored for compact convolutional configurations. This variant is optimized 
for resource-constrained environments while delivering robust performance on 
classification tasks.

**Total Parameters**: 3,774,024

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v4_conv_small(num_classes: int = 1000, **kwargs) -> MobileNet_V4


Parameters
----------
- **num_classes** (*int*, optional):
  Specifies the number of output classes. The default is 1000, 
  commonly used for ImageNet classification.

- **\*\*kwargs** (*dict*):  
  Additional keyword arguments to customize the model configuration. These may override 
  the default settings used in the small convolutional variant.

Usage Example
-------------
.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.mobilenet_v4_conv_small(num_classes=1000)
    >>> print(model)

Details
-------
The `mobilenet_v4_conv_small` function instantiates a MobileNet-v4 model with a configuration optimized for 
efficiency in mobile and embedded systems. By emphasizing a compact convolutional design, it strikes a balance 
between low computational cost and high accuracy. This makes it an excellent choice for applications where 
resources are limited yet performance is critical.

.. note::

   For scenarios requiring different architectural configurations, consider using the base `MobileNet_V4` 
   class with a custom configuration dictionary to better tailor the model to your specific requirements.