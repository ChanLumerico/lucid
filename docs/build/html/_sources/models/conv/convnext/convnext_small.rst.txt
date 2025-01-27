convnext_small
==============

.. autofunction:: lucid.models.convnext_small

The `convnext_small` function creates a ConvNeXt variant with a balanced configuration, 
offering a trade-off between computational efficiency and model capacity. 
This model variant features moderately increased depths and dimensions compared to the 
tiny variant, making it suitable for a wider range of applications.

**Total Parameters**: 46,884,148

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_small(num_classes: int = 1000, **kwargs) -> ConvNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments for further customization of the ConvNeXt model.

Returns
-------

- **ConvNeXt**:
  An instance of the `ConvNeXt` model with a small configuration.

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models import convnext_small

    # Create a ConvNeXt-Small model with default 1000 classes
    model = convnext_small(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)
