convnext_v2_tiny
================

.. autofunction:: lucid.models.convnext_v2_tiny

The `convnext_v2_tiny` creates a ConvNeXt-v2 variant with a compact yet capable 
configuration. It is designed to provide competitive performance across a wide 
range of image classification tasks while remaining efficient in terms of resource usage.

**Total Parameters**: 28,635,496

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_v2_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs**:
  Additional keyword arguments for customizing the ConvNeXt-v2 architecture.

Returns
-------

- **model** (*ConvNeXt_V2*):
  An instance of the ConvNeXt-v2 class configured as the `tiny` variant.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Create convnext_v2_tiny with default 1000 classes
    model = models.convnext_v2_tiny(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)
