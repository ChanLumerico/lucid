mobilenet_v2
============

.. autofunction:: lucid.models.mobilenet_v2

The `mobilenet_v2` function creates a MobileNet-v2 model instance. 
This architecture incorporates inverted residual blocks and linear bottlenecks, 
making it highly efficient for mobile and embedded vision applications.

**Total Parameters**: 3,504,872

Function Signature
------------------

.. code-block:: python

    def mobilenet_v2(num_classes: int = 1000, **kwargs) -> MobileNet_V2

Parameters
----------
- **num_classes** (*int*, optional):
  Specifies the number of output classes for classification. Default is 1000, 
  commonly used for ImageNet.

- **kwargs** (*dict*, optional):
  Additional arguments passed to the `MobileNet_V2` constructor for further customization.

Returns
-------
- **MobileNet_V2**:
  An instance of the MobileNet-v2 model configured with the specified parameters.

Example
-------

.. code-block:: python

    import lucid
    from lucid.models import mobilenet_v2

    # Create a MobileNet-v2 model with default parameters
    model = mobilenet_v2(num_classes=1000)

    # Create a sample input tensor
    input_tensor = lucid.random.randn(1, 3, 224, 224)

    # Perform a forward pass
    output = model(input_tensor)
    print(output)
