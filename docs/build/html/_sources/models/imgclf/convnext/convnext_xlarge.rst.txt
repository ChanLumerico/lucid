convnext_xlarge
===============

.. autofunction:: lucid.models.convnext_xlarge

The `convnext_xlarge` function creates a ConvNeXt variant with an expansive configuration, 
offering the highest model capacity in the ConvNeXt family. This model is optimized for 
large-scale applications and highly complex tasks, featuring the largest depths and 
dimensions for superior performance.

**Total Parameters**: 350,196,968

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_xlarge(num_classes: int = 1000, **kwargs) -> ConvNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments for further customization of the ConvNeXt model.

Returns
-------

- **ConvNeXt**:
  An instance of the `ConvNeXt` model with an extra-large configuration.

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models import convnext_xlarge

    # Create a ConvNeXt-XLarge model with default 1000 classes
    model = convnext_xlarge(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)

    print(output.shape)  # Shape: (1, 1000)

**Custom Number of Classes**

.. code-block:: python

    # Create a ConvNeXt-XLarge model with 21841 classes
    model = convnext_xlarge(num_classes=21841)

    input_ = lucid.random.randn(1, 3, 224, 224)

    output = model(input_)
    print(output.shape)  # Shape: (1, 21841)
