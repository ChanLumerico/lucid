convnext_v2_base
================

.. autofunction:: lucid.models.convnext_v2_base

The `convnext_v2_base` creates a ConvNeXt-v2 variant with a standard configuration, 
designed for robust performance on diverse image classification tasks. This variant 
strikes a balance between scalability and accuracy, making it ideal for most 
general-purpose applications.

**Total Parameters**: 88,717,800

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_v2_base(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs**:
  Additional keyword arguments for customizing the ConvNeXt-v2 architecture.

Returns
-------

- **model** (*ConvNeXt_V2*):
  An instance of the ConvNeXt-v2 class configured as the `base` variant.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Create convnext_v2_base with default 1000 classes
    model = models.convnext_v2_base(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)
