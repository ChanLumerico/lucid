efficientnet_v2_m
=================

.. autofunction:: lucid.models.efficientnet_v2_m

The `efficientnet_v2_m` function instantiates a medium-sized variant of the 
EfficientNet-v2 model, specifically designed for lightweight tasks while maintaining 
high performance.

**Total Parameters**: 55,302,108

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientnet_v2_m(num_classes: int = 1000, **kwargs) -> EfficientNet_V2:

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for the model. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments for configuring the `EfficientNet_V2` model. 
  These parameters are passed directly to the underlying constructor of `EfficientNet_V2`.

Returns
-------

- **EfficientNet_V2**: 
  An instance of the `EfficientNet_V2` model pre-configured for the small variant.

Usage
-----

The `efficientnet_v2_m` function simplifies the creation of an EfficientNet-v2 model for lightweight tasks
such as image classification on smaller datasets. The configuration ensures an optimized balance
between efficiency and accuracy.

Examples
--------

**Creating a EfficientNet-v2-M model**:

.. code-block:: python

    import lucid
    import lucid.models as models

    # Instantiate the medium-sized variant of EfficientNet_V2
    model = models.efficientnet_v2_m(num_classes=100, dropout=0.2)

    # Create a sample input tensor
    input_tensor = lucid.random.randn(1, 3, 224, 224)

    # Perform a forward pass
    output = model(input_tensor)
    print(output)
