coatnet_1
=========

.. autofunction:: lucid.models.coatnet_1

The `coatnet_1` function constructs the CoAtNet-1 preset.
This preset uses the default `CoAtNetConfig` stage layout for the `1` variant.

**Total Parameters**: 53,330,240

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_1(num_classes: int = 1000, **kwargs) -> CoAtNet

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
  A CoAtNet model instance constructed from the `1` preset config.
