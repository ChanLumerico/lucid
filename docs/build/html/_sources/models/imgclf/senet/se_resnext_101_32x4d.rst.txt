se_resnext_101_32x4d
====================

.. autofunction:: lucid.models.se_resnext_101_32x4d

The `se_resnext_101_32x4d` function is a model constructor which creates an 
instance of the SE-ResNeXt-101 (32x4d) architecture, which is a variant of ResNeXt 
that incorporates Squeeze-and-Excitation (SE) blocks for improved performance.

**Total Parameters**: 48,955,416

Function Signature
------------------

.. code-block:: python

    def se_resnext_101_32x4d(num_classes: int = 1000, **kwargs) -> SENet

Parameters
----------
- **num_classes** (*int*, optional):  
  The number of output classes for the model. Default is 1000 (for ImageNet classification).

- **kwargs** (*dict*, optional):  
  Additional keyword arguments to configure the model. 
  This allows flexibility in customizing the model architecture or settings.

Returns
-------
- **SENet**:  
  An instance of the SE-ResNeXt-101 (32x4d) model, ready for training or inference.

Key Features
------------
- **ResNeXt Architecture**:  
  Incorporates grouped convolutions to improve computational efficiency while 
  maintaining model capacity.

- **Squeeze-and-Excitation (SE) Blocks**:  
  Enhances feature representations by adaptively recalibrating channel-wise responses.

- **Flexible Output Classes**:  
  Can be configured for different classification tasks by adjusting the `num_classes` 
  parameter.

Example Usage
-------------

The following example demonstrates how to instantiate the `se_resnext_101_32x4d` 
model for a custom task:

.. code-block:: python

    from lucid.nn.models import se_resnext_101_32x4d

    # Create the model for a custom task with 1000 classes
    model = se_resnext_101_32x4d(num_classes=1000)

    # Print the model summary
    print(model)

    # Forward pass with a sample input
    import lucid
    input_tensor = lucid.random.randn(1, 3, 224, 224)  # Batch of one image, 3 channels, 224x224 resolution
    output = model(input_tensor)

    print("Output shape:", output.shape)
