sk_resnext_50_32x4d
===================

.. autofunction:: lucid.models.sk_resnext_50_32x4d

The `sk_resnext_50_32x4d` function is a model constructor which creates an 
instance of the SK-ResNeXt-50 (32x4d) architecture, which is a variant of 
ResNeXt enhanced with Selective Kernel (SK) mechanisms for dynamic kernel selection.

**Total Parameters**: 29,274,760

Function Signature
------------------

.. code-block:: python

    def sk_resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> SKNet

Parameters
----------
- **num_classes** (*int*, optional):  
  The number of output classes for the model. Default is 1000 (for ImageNet classification).

- **kwargs** (*dict*, optional):  
  Additional keyword arguments to configure the model. 
  This allows flexibility in customizing the model architecture or settings.

Returns
-------
- **SKNet**:  
  An instance of the SK-ResNeXt-50 (32x4d) model, ready for training or inference.

Key Features
------------
- **ResNeXt Architecture**:  
  Incorporates grouped convolutions for efficient computation while maintaining 
  high model capacity.

- **Selective Kernel (SK) Blocks**:  
  Dynamically adjusts receptive fields by adaptively combining multiple kernels, 
  enhancing feature representation.

- **Flexible Output Classes**:  
  Configurable for different classification tasks by adjusting the `num_classes` parameter.

Example Usage
-------------

The following example demonstrates how to instantiate the `sk_resnext_50_32x4d` 
model for a custom task:

.. code-block:: python

    from lucid.nn.models import sk_resnext_50_32x4d

    # Create the model for a custom task with 1000 classes
    model = sk_resnext_50_32x4d(num_classes=1000)

    # Print the model summary
    print(model)

    # Forward pass with a sample input
    import lucid
    input_tensor = lucid.random.randn(1, 3, 224, 224)  # Batch of one image, 3 channels, 224x224 resolution
    output = model(input_tensor)

    print("Output shape:", output.shape)
