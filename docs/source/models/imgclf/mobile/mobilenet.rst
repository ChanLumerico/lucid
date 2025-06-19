mobilenet
=========

.. autofunction:: lucid.models.mobilenet

The `mobilenet` function creates a MobileNet model instance. It supports customizing 
the width multiplier and the number of output classes for flexibility across various 
use cases.

**Total Parameters**: 4,232,008

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet(width_multiplier: float = 1.0, num_classes: int = 1000, **kwargs) -> MobileNet

Parameters
----------
- **width_multiplier** (*float*, optional):
  Scales the width of the network by adjusting the number of channels in each layer. 
  Default is 1.0, which corresponds to the full model size.

- **num_classes** (*int*, optional):
  Specifies the number of output classes for classification. Default is 1000, 
  commonly used for ImageNet.

- **kwargs** (*dict*, optional):
  Additional arguments passed to the `MobileNet` constructor for further customization.

Returns
-------
- **MobileNet**:
  An instance of the MobileNet model configured with the specified parameters.

Example
-------

.. code-block:: python

    import lucid
    from lucid.models import mobilenet

    # Create a MobileNet model with default parameters
    model = mobilenet(width_multiplier=1.0, num_classes=1000)

    # Create a sample input tensor
    input_tensor = lucid.random.randn(1, 3, 224, 224)

    # Perform a forward pass
    output = model(input_tensor)
    print(output)
