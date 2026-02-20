inception_next_base
===================

.. autofunction:: lucid.models.inception_next_base

The `inception_next_base` function instantiates an InceptionNeXt-Base model,
a standard variant of the InceptionNeXt architecture optimized for high accuracy
and robust feature extraction in deep learning tasks.

**Total Parameters**: 86,748,840

Function Signature
------------------

.. code-block:: python

    @register_model
    def inception_next_base(num_classes: int = 1000, **kwargs) -> InceptionNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the final classification layer. Defaults to 1000.

- **kwargs** (*dict*, optional):
  Additional arguments to customize the model configuration.

Returns
-------

- **InceptionNeXt**:
  An instance of the `InceptionNeXt` model configured as the small variant.

Example
-------

.. code-block:: python

    from lucid.models import inception_next_base

    # Instantiate InceptionNeXt-Small
    model = inception_next_base(num_classes=1000)

    # Forward pass with a random input
    input_tensor = lucid.random.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)
