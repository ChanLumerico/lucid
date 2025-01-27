convnext_tiny
=============

.. autofunction:: lucid.models.convnext_tiny

The `convnext_tiny` creates a ConvNeXt variant with a compact configuration, 
optimized for efficient computation and suitable for scenarios with limited 
computational resources. This model variant is designed with smaller depths 
and dimensions compared to the base ConvNeXt architecture.

**Total Parameters**: 28,589,128

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments for further customization of the ConvNeXt model.

Returns
-------

- **ConvNeXt**:
  An instance of the `ConvNeXt` model with a tiny configuration.

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models import convnext_tiny

    # Create a ConvNeXt-Tiny model with default 1000 classes
    model = convnext_tiny(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)
