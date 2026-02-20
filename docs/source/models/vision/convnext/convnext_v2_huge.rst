convnext_v2_huge
================

.. autofunction:: lucid.models.convnext_v2_huge

The `convnext_v2_huge` creates a ConvNeXt-v2 variant with a maximal configuration, 
optimized for state-of-the-art performance on large-scale and high-complexity 
image classification tasks. This model is designed for applications requiring 
abundant computational resources and top-tier accuracy.

**Total Parameters**: 660,289,640

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_v2_huge(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs**:
  Additional keyword arguments for customizing the ConvNeXt-v2 architecture.

Returns
-------

- **model** (*ConvNeXt_V2*):
  An instance of the ConvNeXt-v2 class configured as the `huge` variant.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Create convnext_v2_huge with default 21841 classes
    model = models.convnext_v2_huge(num_classes=21841)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 21841)
