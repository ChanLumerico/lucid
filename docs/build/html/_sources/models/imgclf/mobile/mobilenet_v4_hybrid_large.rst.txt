mobilenet_v4_hybrid_large
=========================

.. autofunction:: lucid.models.mobilenet_v4_hybrid_large

Overview
--------
The `mobilenet_v4_hybrid_large` function instantiates a `MobileNet_V4` model variant that leverages a 
hybrid architecture in its large configuration. This model is designed to maximize performance and accuracy 
for demanding classification tasks by integrating advanced convolutional strategies with supplementary architectural 
enhancements.

**Total Parameters**: 37,755,152

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v4_hybrid_large(num_classes: int = 1000, **kwargs) -> MobileNet_V4

Parameters
----------
- **num_classes** (*int*, optional):  
  Specifies the number of output classes. The default value is 1000, aligning with standard benchmarks like 
  ImageNet.

- **\*\*kwargs** (*dict*):  
  Additional keyword arguments for further customization of the model. These can override default configurations 
  to adapt the network for specific scenarios.

Usage Example
-------------
.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.mobilenet_v4_hybrid_large(num_classes=1000)
    >>> print(model)

Details
-------
The `mobilenet_v4_hybrid_large` function delivers a high-performance MobileNet-v4 model by employing a large-scale 
hybrid architecture that integrates cutting-edge convolutional techniques with extra architectural optimizations. 
This design is optimal for scenarios where the highest accuracy is required and computational resources are abundant.

.. note::

   Due to its increased complexity and resource demands, the hybrid large variant is best suited for applications 
   where performance is prioritized over computational efficiency.
