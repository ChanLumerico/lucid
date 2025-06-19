resnext_50_32x4d
================

.. autofunction:: lucid.models.resnext_50_32x4d

The `resnext_50_32x4d` function constructs a ResNeXt-50 model with a configuration 
of 32 groups and a base width of 4. 
This implementation is based on the `ResNeXt` class, leveraging grouped convolutions 
to increase model capacity while maintaining efficiency.

**Total Parameters**: 25,028,904

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> ResNeXt

Parameters
----------
- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default: 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments passed to the `ResNeXt` class.

Returns
-------
- **ResNeXt**:
  A ResNeXt-50 model instance with 32 groups and a base width of 4.

Description
-----------
The `resnext_50_32x4d` function initializes a ResNeXt-50-like architecture. 
The model configuration includes:

- **Layers**: [3, 4, 6, 3] stages, each containing a specified number of blocks.
- **Cardinality**: 32 groups for grouped convolutions.
- **Base Width**: Feature width of 4 channels per group.

This setup achieves a balance between representational capacity and computational efficiency.

Examples
--------

**Basic Example**:

.. code-block:: python

    >>> from lucid.models import resnext_50_32x4d
    >>> model = resnext_50_32x4d(num_classes=1000)
    >>> input_tensor = Tensor(np.random.randn(8, 3, 224, 224))  # Shape: (N, C, H, W)
    >>> output = model(input_tensor)  # Forward pass
    >>> print(output.shape)
    (8, 1000)

.. note::

  - The `resnext_50_32x4d` function is registered under the model registry 
    for easy access through the `@register_model` decorator.
  - This model is particularly suitable for tasks requiring efficient and 
    scalable deep learning architectures.
