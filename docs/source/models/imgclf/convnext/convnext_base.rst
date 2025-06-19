convnext_base
=============

.. autofunction:: lucid.models.convnext_base

The `convnext_base` function creates a ConvNeXt variant with a standard configuration, 
providing a robust model capacity suitable for general-purpose applications. 
This variant has larger depths and dimensions, offering enhanced performance on 
image classification tasks.

**Total Parameters**: 88,591,464

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_base(num_classes: int = 1000, **kwargs) -> ConvNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments for further customization of the ConvNeXt model.

Returns
-------

- **ConvNeXt**:
  An instance of the `ConvNeXt` model with a base configuration.

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models import convnext_base

    # Create a ConvNeXt-Base model with default 1000 classes
    model = convnext_base(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)
