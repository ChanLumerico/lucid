densenet_201
============

.. autofunction:: lucid.models.densenet_201

The `densenet_201` function constructs a DenseNet-201 model, 
a specific variant of the DenseNet architecture.

It is configured with four dense blocks, following the layer configuration: (6, 12, 48, 32).
This model is well-suited for image classification tasks.

**Total Parameters**: 20,013,928

Function Signature
------------------

.. code-block:: python

    def densenet_201(num_classes: int = 1000, **kwargs) -> DenseNet

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
  An instance of the `DenseNet` class configured as DenseNet-201.

Examples
--------

**Creating a DenseNet-201 model for ImageNet classification:**

.. code-block:: python

    from lucid.models import densenet_201

    model = densenet_201(num_classes=1000)

    input_tensor = lucid.random.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)
