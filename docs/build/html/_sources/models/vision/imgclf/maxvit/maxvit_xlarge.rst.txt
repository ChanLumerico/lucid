maxvit_xlarge
=============

.. autofunction:: lucid.models.maxvit_xlarge

The `maxvit_xlarge` function constructs the xlarge MaxViT preset.

Function Signature
------------------

.. code-block:: python

    @register_model
    def maxvit_xlarge(in_channels: int = 3, num_classes: int = 1000, **kwargs) -> MaxViT

Parameters
----------

- **in_channels** (*int*, optional):
  Number of input channels. Default is `3`.
- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `MaxViTConfig`, excluding the preset
  `depths`, `channels`, and `embed_dim` fields.

Returns
-------

- **MaxViT**:
  A `MaxViT` instance constructed from the xlarge preset config.

Preset Config
-------------

- **depths**: `(2, 6, 14, 2)`
- **channels**: `(192, 384, 768, 1536)`
- **embed_dim**: `192`
