convnext_v2_pico
================

.. autofunction:: lucid.models.convnext_v2_pico

The `convnext_v2_pico` creates a ConvNeXt-v2 variant with a lightweight configuration, 
optimized for efficient computation and performance. This variant is well-suited for 
scenarios with limited computational resources while still targeting competitive accuracy 
on medium-scale image classification tasks.

**Total Parameters**: 9,066,280

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_v2_pico(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs**:
  Additional keyword arguments for customizing the ConvNeXt-v2 architecture.

Returns
-------

- **model** (*ConvNeXt_V2*):
  An instance of the ConvNeXt-v2 class configured as the `pico` variant.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Create convnext_v2_pico with default 1000 classes
    model = models.convnext_v2_pico(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)
