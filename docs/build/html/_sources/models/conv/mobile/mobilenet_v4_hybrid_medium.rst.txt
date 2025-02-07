mobilenet_v4_hybrid_medium
==========================

.. autofunction:: lucid.models.mobilenet_v4_hybrid_medium

Overview
--------
The `mobilenet_v4_hybrid_medium` function returns an instance of the `MobileNet_V4` model featuring a 
hybrid architecture that blends traditional convolutional layers with innovative enhancements. This medium variant 
is optimized to balance computational efficiency with improved accuracy, making it suitable for a wide range of 
applications.

**Total Parameters**: 11,070,136

Parameters
----------
- **num_classes** (*int*, optional):  
  Defines the number of output classes. The default is 1000, commonly used for large-scale classification tasks.

- **\*\*kwargs** (*dict*):  
  Additional keyword arguments that allow customization of the hybrid configuration. These parameters can be used 
  to fine-tune the model's architecture based on specific requirements.

Usage Example
-------------
.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.mobilenet_v4_hybrid_medium(num_classes=1000)
    >>> print(model)

Details
-------
The `mobilenet_v4_hybrid_medium` function constructs a MobileNet-v4 model with a hybrid architecture that 
combines the benefits of convolutional layers with enhanced module designs. It provides a compromise between 
accuracy and computational cost, making it a versatile choice for various deployment scenarios.

.. note::

   This medium hybrid variant is ideal for users seeking a balance between performance and resource consumption. 
   For specific needs, consider adjusting the configuration parameters via the additional keyword arguments.
