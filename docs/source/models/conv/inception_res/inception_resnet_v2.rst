inception_resnet_v2
===================

.. autofunction:: lucid.models.inception_resnet_v2

Overview
--------

The `inception_resnet_v2` function implements the Inception-ResNet v2 architecture, 
which builds on Inception-ResNet v1 with deeper layers and improved efficiency. 
This model leverages the advantages of Inception modules and residual connections 
for enhanced performance on complex image classification tasks.

This function returns a preconfigured `InceptionResNet` model optimized for use in 
various applications, while allowing for flexibility in the number of output classes 
and other customizations.

**Total Parameters**: 35,847,512

Function Signature
------------------

.. code-block:: python

   @register_model
   def inception_resnet_v2(
       num_classes: int = 1000,
       **kwargs,
   ) -> InceptionResNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for the final classification layer. Default is `1000`.

- **kwargs** (*dict*, optional):
  Additional arguments passed to the underlying `InceptionResNet` base class or components.

Returns
-------

- **InceptionResNet**:
  An instance of the `InceptionResNet` model configured for the v2 architecture.

Example Usage
-------------

Below is an example of defining and using the `inception_resnet_v2` function:

.. code-block:: python

   import lucid.models as models

   # Create an Inception-ResNet v2 model with default parameters
   model = models.inception_resnet_v2(num_classes=1000)

   # Sample input tensor (e.g., batch of 299x299 RGB images)
   input_tensor = lucid.Tensor([...])

   # Forward pass
   output = model(input_tensor)
   print(output)
