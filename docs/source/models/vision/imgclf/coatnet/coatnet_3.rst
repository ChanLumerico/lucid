coatnet_3
=========

.. autofunction:: lucid.models.coatnet_3

The `coatnet_3` function constructs the CoAtNet-3 preset.
This preset uses the default `CoAtNetConfig` stage layout for the `3` variant.

**Total Parameters**: 157,790,656

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_3(num_classes: int = 1000, **kwargs) -> CoAtNet

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `CoAtNetConfig`, excluding the
  preset `img_size`, `in_channels`, `num_blocks`, and `channels` fields.

Returns
-------

- **CoAtNet**:
  A CoAtNet model instance constructed from the `3` preset config.
