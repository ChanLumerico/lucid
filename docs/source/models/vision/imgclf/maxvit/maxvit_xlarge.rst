maxvit_xlarge
=============

.. autofunction:: lucid.models.maxvit_xlarge

The `maxvit_xlarge` function returns the highest capacity variant of the MaxViT architecture.  
It significantly increases both depth and channel width, making it ideal 
for high-resolution datasets and demanding image classification tasks.

**Total Parameters**: 383,734,024

Function Signature
------------------

.. code-block:: python

    def maxvit_xlarge(
        in_channels: int = 3,
        num_classes: int = 1000,
        **kwargs
    ) -> MaxViT

Parameters
----------

- **in_channels** (*int*, optional):  
  Number of input channels (e.g., `3` for RGB images). Default is `3`.

- **num_classes** (*int*, optional):  
  Number of output classes for classification. Default is `1000`.

- **kwargs** (*any*, optional):  
  Additional keyword arguments passed to the :class:`MaxViT` constructor.

Model Configuration
-------------------

This `xlarge` variant is configured as:

- **depths**: `(2, 6, 14, 2)`  
- **channels**: `(192, 384, 768, 1536)`  
- **embed_dim**: `192`

These values offer maximum feature capacity, especially suitable for large-scale datasets.

Example
-------

.. code-block:: python

    import lucid
    from lucid.models.transformer import maxvit_xlarge

    model = maxvit_xlarge()
    input_tensor = lucid.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # (1, 1000)

.. warning::
    Due to its large number of parameters, this model is recommended for 
    use on powerful hardware (e.g., GPU or Apple Silicon with MLX backend).
