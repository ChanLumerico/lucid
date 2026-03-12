coatnet_7
=========

.. autofunction:: lucid.models.coatnet_7

The `coatnet_7` function constructs the CoAtNet-7 preset.
This preset uses the default `CoAtNetConfig` stage layout for the `7` variant.

**Total Parameters**: 3,107,978,688

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_7(num_classes: int = 1000, **kwargs) -> CoAtNet

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
  A CoAtNet model instance constructed from the `7` preset config.
