coatnet_4
=========

.. autofunction:: lucid.models.coatnet_4

The `coatnet_4` function constructs the CoAtNet-4 preset.
This preset uses the default `CoAtNetConfig` stage layout for the `4` variant.

**Total Parameters**: 277,301,632

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_4(num_classes: int = 1000, **kwargs) -> CoAtNet

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
  A CoAtNet model instance constructed from the `4` preset config.
