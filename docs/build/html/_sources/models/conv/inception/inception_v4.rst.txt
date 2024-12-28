models.inception_v4
===================

.. autofunction:: lucid.models.inception_v4

Overview
--------

The `inception_v4` function implements the Inception v4 architecture, which combines
the improved features of Inception v3 with residual connections to enhance gradient flow and optimization.
This model excels in image classification tasks, leveraging deeper networks and improved module designs.

**Total Parameters**: 40,586,984

Function Signature
------------------

.. code-block:: python

   @register_model
   def inception_v4(num_classes: int = 1000, **kwargs) -> Inception

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for the final classification layer. Default is `1000`.

- **kwargs** (*dict*, optional):
  Additional arguments passed to the underlying `Inception` base class or model components.

Returns
-------

- **Inception**:
  An instance of the `Inception` model configured for the v4 architecture.

Example Usage
-------------

Below is an example of defining and using the Inception v4 model:

.. code-block:: python

   import lucid.models as models

   # Create an Inception v4 model with default parameters
   model = models.inception_v4(num_classes=1000)

   # Sample input tensor (e.g., batch of 299x299 RGB images)
   input_tensor = lucid.Tensor([...])

   # Forward pass
   output = model(input_tensor)
   print(output)
