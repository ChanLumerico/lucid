models.inception_v1
===================

.. autofunction:: lucid.models.inception_v1

Overview
--------

The `inception_v1` function provides an implementation of the Inception v1 (GoogLeNet) architecture.
This model was introduced in the paper "Going Deeper with Convolutions" by Szegedy et al.
It is designed for image classification tasks and includes the option to use auxiliary 
classifiers to improve optimization during training.

**Total Parameters**: 13,393,352

Function Signature
------------------

.. code-block:: python

   @register_model
   def inception_v1(
       num_classes: int = 1000,
       use_aux: bool = True,
       **kwargs,
   ) -> Inception

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for the final classification layer. Default is `1000`.

- **use_aux** (*bool*, optional):
  Whether to include auxiliary classifiers. 
  Auxiliary classifiers are additional branches used during training to assist optimization. 
  Default is `True`.

- **kwargs** (*dict*, optional):
  Additional arguments passed to the underlying `Inception` base class or model components.

Returns
-------

- **Inception**:
  An instance of the `Inception` model configured for the v1 architecture.

Example Usage
-------------

Below is an example of defining and using the Inception v1 model:

.. code-block:: python

   import lucid.models as models

   # Create an Inception v1 model with default parameters
   model = models.inception_v1(num_classes=1000, use_aux=True)

   # Sample input tensor (e.g., batch of 224x224 RGB images)
   input_tensor = lucid.Tensor([...])

   # Forward pass
   output = model(input_tensor)
   print(output)
