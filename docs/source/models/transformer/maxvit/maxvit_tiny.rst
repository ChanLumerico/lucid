maxvit_tiny
===========

.. autofunction:: lucid.models.maxvit_tiny

The `maxvit_tiny` function returns a preconfigured instance of the `MaxViT` 
architecture with a lightweight "tiny" setup. This configuration is suitable for 
tasks requiring a small model footprint with effective performance on visual tasks.

Total Parameters: 25,081,416

Function Signature
------------------

.. code-block:: python

    def maxvit_tiny(
        in_channels: int = 3,
        num_classes: int = 1000,
        **kwargs
    ) -> MaxViT

Parameters
----------

- **in_channels** (*int*, optional):  
  Number of channels in the input image. Default is `3` (RGB images).

- **num_classes** (*int*, optional):  
  Number of output classes for classification. Default is `1000`.

- **kwargs** (*any*, optional):  
  Additional keyword arguments passed to the `MaxViT` constructor.

Model Configuration
-------------------

This preset initializes a `MaxViT` model with the following configuration:

- **depths**: `(2, 2, 5, 2)` — Number of blocks per stage.
- **channels**: `(64, 128, 256, 512)` — Channel widths per stage.
- **embed_dim**: `64` — Initial embedding dimension.

Example
-------

.. code-block:: python

    import lucid
    from lucid.models.transformer import maxvit_tiny

    model = maxvit_tiny()
    input_tensor = lucid.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # (1, 1000)

.. tip::
    Use `**kwargs` to override components such as activation, norm layers, or dropout.
