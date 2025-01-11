densenet_264
============

.. autofunction:: lucid.models.densenet_264

The `densenet_264` function constructs a DenseNet-264 model, 
a specific variant of the DenseNet architecture.

It is configured with four dense blocks, following the layer configuration: (6, 12, 64, 48).
This model is well-suited for image classification tasks.

**Total Parameters**: 33,337,704

Function Signature
------------------

.. code-block:: python

    def densenet_264(num_classes: int = 1000, **kwargs) -> DenseNet

Parameters
----------
- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.

- **kwargs** (*dict*):
  Additional keyword arguments passed to the `DenseNet` constructor, 
  such as `growth_rate` and `num_init_features`.

Returns
-------
- **DenseNet**:
  An instance of the `DenseNet` class configured as DenseNet-264.

Examples
--------

**Creating a DenseNet-264 model for ImageNet classification:**

.. code-block:: python

    from lucid.models import densenet_264

    model = densenet_264(num_classes=1000)

    input_tensor = lucid.random.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)
