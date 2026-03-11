maxvit_base
===========

.. autofunction:: lucid.models.maxvit_base

The `maxvit_base` function constructs the base MaxViT preset.

Function Signature
------------------

.. code-block:: python

    @register_model
    def maxvit_base(in_channels: int = 3, num_classes: int = 1000, **kwargs) -> MaxViT

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
  A `MaxViT` instance constructed from the base preset config.

Preset Config
-------------

- **depths**: `(2, 6, 14, 2)`
- **channels**: `(96, 192, 384, 768)`
- **embed_dim**: `64`
