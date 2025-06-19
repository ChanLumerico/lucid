inception_next_atto
===================

.. autofunction:: lucid.models.inception_next_atto

The `inception_next_atto` function instantiates an InceptionNeXt-Atto model,
a lightweight variant of the InceptionNeXt architecture designed for
efficient performance with minimal computational resources.

**Total Parameters**: 4,156,520

Function Signature
------------------

.. code-block:: python

    @register_model
    def inception_next_atto(num_classes: int = 1000, **kwargs) -> InceptionNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the final classification layer. Defaults to 1000.

- **kwargs** (*dict*, optional):
  Additional arguments to customize the model configuration.

Returns
-------

- **InceptionNeXt**:
  An instance of the `InceptionNeXt` model configured as the Atto variant.

Example
-------

.. code-block:: python

    from lucid.models import inception_next_atto

    # Instantiate InceptionNeXt-Atto
    model = inception_next_atto(num_classes=1000)

    # Forward pass with a random input
    input_tensor = lucid.random.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)
