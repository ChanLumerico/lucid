xception
========

.. autofunction:: lucid.models.xception

The `xception` function constructs the standard Xception classification model.
This preset uses the default `XceptionConfig` architecture.

**Total Parameters**: 22,862,096

Function Signature
------------------

.. code-block:: python

    @register_model
    def xception(num_classes: int = 1000, **kwargs) -> Xception

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is 1000.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `XceptionConfig`, excluding the preset
  `stem_channels`, `entry_channels`, `middle_channels`, `middle_repeats`, and `exit_channels` fields.

Returns
-------

- **Xception**:
  An Xception model instance configured from the default architecture preset.
