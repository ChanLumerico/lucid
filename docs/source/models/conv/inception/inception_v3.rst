models.inception_v3
===================

.. autofunction:: lucid.models.inception_v3

Overview
--------

The `inception_v3` function provides an implementation of the Inception v3 architecture.
This model improves upon the original Inception design by introducing factorized convolutions,
auxiliary classifiers, and label smoothing, among other enhancements. 
It is well-suited for image classification tasks.

**Total Parameters**: 30,817,392

Function Signature
------------------

.. code-block:: python

   @register_model
   def inception_v3(
       num_classes: int = 1000,
       use_aux: bool = True,
       dropout_prob: float = 0.5,
       **kwargs,
   ) -> Inception

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for the final classification layer. Default is `1000`.

- **use_aux** (*bool*, optional):
  Whether to include auxiliary classifiers. Auxiliary classifiers are additional branches 
  used during training to assist optimization. Default is `True`.

- **dropout_prob** (*float*, optional):
  The probability of an element to be zeroed during dropout. This helps regularize the model. 
  Default is `0.5`.

- **kwargs** (*dict*, optional):
  Additional arguments passed to the underlying `Inception` base class or model components.

Returns
-------

- **Inception**:
  An instance of the `Inception` model configured for the v3 architecture.

Example Usage
-------------

Below is an example of defining and using the Inception v3 model:

.. code-block:: python

   import lucid.models as models

   # Create an Inception v3 model with default parameters
   model = models.inception_v3(num_classes=1000, use_aux=True, dropout_prob=0.5)

   # Sample input tensor (e.g., batch of 299x299 RGB images)
   input_tensor = lucid.Tensor([...])

   # Forward pass
   output = model(input_tensor)
   print(output)
