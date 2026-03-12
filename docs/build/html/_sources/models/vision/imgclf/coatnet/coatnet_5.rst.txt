coatnet_5
=========

.. autofunction:: lucid.models.coatnet_5

The `coatnet_5` function constructs the CoAtNet-5 preset.
This preset uses the default `CoAtNetConfig` stage layout for the `5` variant.

**Total Parameters**: 770,124,608

Function Signature
------------------

.. code-block:: python

    @register_model
    def coatnet_5(num_classes: int = 1000, **kwargs) -> CoAtNet

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `CoAtNetConfig`, excluding the
  preset `img_size`, `in_channels`, `num_blocks`, `channels`, and `num_heads` fields.

Returns
-------

- **CoAtNet**:
  A CoAtNet model instance constructed from the `5` preset config.
