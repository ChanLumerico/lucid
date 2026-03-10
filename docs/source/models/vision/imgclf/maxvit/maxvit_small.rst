maxvit_small
============

.. autofunction:: lucid.models.maxvit_small

The `maxvit_small` function constructs the small MaxViT preset.

Function Signature
------------------

.. code-block:: python

    @register_model
    def maxvit_small(in_channels: int = 3, num_classes: int = 1000, **kwargs) -> MaxViT

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
  A `MaxViT` instance constructed from the small preset config.

Preset Config
-------------

- **depths**: `(2, 2, 5, 2)`
- **channels**: `(96, 192, 384, 768)`
- **embed_dim**: `64`
