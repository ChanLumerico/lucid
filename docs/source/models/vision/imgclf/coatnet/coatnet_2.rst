coatnet_2
=========

.. autofunction:: lucid.models.coatnet_2

The `coatnet_2` function constructs the CoAtNet-2 preset.
This preset uses the default `CoAtNetConfig` stage layout for the `2` variant.

**Total Parameters**: 82,516,096

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_2(num_classes: int = 1000, **kwargs) -> CoAtNet

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
  A CoAtNet model instance constructed from the `2` preset config.
