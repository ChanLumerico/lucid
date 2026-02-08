convnext_large
==============

.. autofunction:: lucid.models.convnext_large

The `convnext_large` function creates a ConvNeXt variant with a high-capacity configuration, 
designed for tasks requiring significant model expressiveness. This variant features 
increased depths and dimensions, making it ideal for more complex applications and datasets.

**Total Parameters**: 197,767,336

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_large(num_classes: int = 1000, **kwargs) -> ConvNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments for further customization of the ConvNeXt model.

Returns
-------

- **ConvNeXt**:
  An instance of the `ConvNeXt` model with a large configuration.

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models import convnext_large

    # Create a ConvNeXt-Large model with default 1000 classes
    model = convnext_large(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)
