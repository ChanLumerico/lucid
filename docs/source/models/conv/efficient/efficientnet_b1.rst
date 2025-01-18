efficientnet_b1
===============

.. autofunction:: lucid.models.efficientnet_b1

The `efficientnet_b1` function creates an instance of the `EfficientNet-B1` model, 
a lightweight and efficient convolutional neural network preconfigured with parameters 
suitable for EfficientNet-B1.

**Total Parameters**: 7,795,560

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientnet_b1(num_classes: int = 1000, **kwargs) -> EfficientNet:

Parameters
----------

- **num_classes** (*int*, optional):
   The number of output classes for the classification task. 
   Defaults to 1000 (e.g., for ImageNet).

- **kwargs** (*dict*, optional):
   Additional keyword arguments passed to the `EfficientNet` class for further customization.

Returns
-------

- **EfficientNet**:
   An instance of the `EfficientNet` class configured with parameters for EfficientNet-B1.

Examples
--------

.. code-block:: python

    from lucid.models import efficientnet_b1

    # Create an EfficientNet-B1 model
    model = efficientnet_b1(num_classes=1000)

    # Forward pass with a random input
    input_tensor = lucid.random.randn(1, 3, 224, 224)  # Batch size of 1, ImageNet resolution
    output = model(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)
