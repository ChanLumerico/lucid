inception_next_tiny
===================

.. autofunction:: lucid.models.inception_next_tiny

The `inception_next_tiny` function instantiates an InceptionNeXt-Tiny model,
a compact variant of the InceptionNeXt architecture designed for a balance
between efficiency and accuracy with moderate computational requirements.

**Total Parameters**: 28,083,832

Function Signature
------------------

.. code-block:: python

    @register_model
    def inception_next_tiny(num_classes: int = 1000, **kwargs) -> InceptionNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the final classification layer. Defaults to 1000.

- **kwargs** (*dict*, optional):
  Additional arguments to customize the model configuration.

Returns
-------

- **InceptionNeXt**:
  An instance of the `InceptionNeXt` model configured as the tiny variant.

Example
-------

.. code-block:: python

    from lucid.models import inception_next_tiny

    # Instantiate InceptionNeXt-Tiny
    model = inception_next_tiny(num_classes=1000)

    # Forward pass with a random input
    input_tensor = lucid.random.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)
