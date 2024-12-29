models.inception_resnet_v1
==========================

.. autofunction:: lucid.models.inception_resnet_v1

Overview
--------

The `inception_resnet_v1` function implements the Inception-ResNet v1 architecture. 
This model combines the multi-scale processing of Inception modules with the 
residual connections of ResNet, enabling efficient optimization and enhanced 
performance for image classification tasks. 

This function returns a preconfigured `InceptionResNet` model for use in 
various applications, with the flexibility to adjust the number of output classes 
or include additional customizations.

**Total Parameters**: 22,739,128

Function Signature
------------------

.. code-block:: python

   @register_model
   def inception_resnet_v1(num_classes: int = 1000, **kwargs) -> InceptionResNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for the final classification layer. Default is `1000`.

- **kwargs** (*dict*, optional):
  Additional arguments passed to the underlying `InceptionResNet` base class or components.

Returns
-------

- **InceptionResNet**:
  An instance of the `InceptionResNet` model configured for the v1 architecture.

Example Usage
-------------

Below is an example of defining and using the `inception_resnet_v1` function:

.. code-block:: python

   import lucid.models as models

   # Create an Inception-ResNet v1 model with default parameters
   model = models.inception_resnet_v1(num_classes=1000)

   # Sample input tensor (e.g., batch of 299x299 RGB images)
   input_tensor = lucid.Tensor([...])

   # Forward pass
   output = model(input_tensor)
   print(output)
