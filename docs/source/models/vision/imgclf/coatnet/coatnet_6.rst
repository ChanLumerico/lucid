coatnet_6
=========

.. autofunction:: lucid.models.coatnet_6

The `coatnet_6` function constructs the CoAtNet-6 preset.
This preset uses the default `CoAtNetConfig` stage layout for the `6` variant.

**Total Parameters**: 2,011,558,336

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_6(num_classes: int = 1000, **kwargs) -> CoAtNet

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `CoAtNetConfig`, excluding the
  preset `img_size`, `in_channels`, `num_blocks`, `channels`, `num_heads`,
  `scaled_num_blocks`, and `scaled_channels` fields.

Returns
-------

- **CoAtNet**:
  A CoAtNet model instance constructed from the `6` preset config.
