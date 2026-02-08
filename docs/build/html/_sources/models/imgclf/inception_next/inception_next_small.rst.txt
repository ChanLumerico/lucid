inception_next_small
====================

.. autofunction:: lucid.models.inception_next_small

The `inception_next_small` function instantiates an InceptionNeXt-Small model,
a mid-sized variant of the InceptionNeXt architecture offering improved performance
while maintaining efficiency for practical deep learning applications.

**Total Parameters**: 49,431,544

Function Signature
------------------

.. code-block:: python

    @register_model
    def inception_next_small(num_classes: int = 1000, **kwargs) -> InceptionNeXt

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

    from lucid.models import inception_next_small

    # Instantiate InceptionNeXt-Small
    model = inception_next_small(num_classes=1000)

    # Forward pass with a random input
    input_tensor = lucid.random.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)
