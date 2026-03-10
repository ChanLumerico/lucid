coatnet_0
=========

.. autofunction:: lucid.models.coatnet_0

The `coatnet_0` function constructs the CoAtNet-0 preset.
This preset uses the default `CoAtNetConfig` stage layout for the `0` variant.

**Total Parameters**: 27,174,944

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_0(num_classes: int = 1000, **kwargs) -> CoAtNet

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
  A CoAtNet model instance constructed from the `0` preset config.
