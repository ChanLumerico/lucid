mobilenet_v4_conv_large
=======================

.. autofunction:: lucid.models.mobilenet_v4_conv_large

Overview
--------
The `mobilenet_v4_conv_large` function instantiates a `MobileNet_V4` model configured for a large-scale 
convolutional variant. This variant is designed for higher accuracy requirements and is suited for devices 
with ample computational resources.

**Total Parameters**: 32,590,864

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v4_conv_large(num_classes: int = 1000, **kwargs) -> MobileNet_V4

Parameters
----------
- **num_classes** (*int*, optional):  
  Specifies the number of output classes. The default is 1000, which is standard for datasets such as ImageNet.

- **\*\*kwargs** (*dict*):  
  Additional keyword arguments to customize the model configuration. These can be used to modify default settings 
  and adapt the architecture for specific tasks.

Usage Example
-------------
.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.mobilenet_v4_conv_large(num_classes=1000)
    >>> print(model)

Details
-------
The `mobilenet_v4_conv_large` function leverages a large convolutional configuration to boost model accuracy by 
increasing the depth and capacity of the network.

.. note::

   For applications where computational resources are limited, 
   consider using the smaller or medium variants.
