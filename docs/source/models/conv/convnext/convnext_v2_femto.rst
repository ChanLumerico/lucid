convnext_v2_femto
=================

.. autofunction:: lucid.models.convnext_v2_femto

The `convnext_v2_femto` creates a ConvNeXt-v2 variant with a very compact configuration, 
designed for moderately resource-constrained environments. This model variant balances 
computational efficiency and performance, making it suitable for small-to-medium-scale 
image classification tasks.

**Total Parameters**: 5,233,240

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_v2_femto(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs**:
  Additional keyword arguments for customizing the ConvNeXt-v2 architecture.

Returns
-------

- **model** (*ConvNeXt_V2*):
  An instance of the ConvNeXt-v2 class configured as the `femto` variant.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Create convnext_v2_femto with default 1000 classes
    model = models.convnext_v2_femto(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)
