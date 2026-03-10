densenet_264
============

.. autofunction:: lucid.models.densenet_264

The `densenet_264` function constructs a DenseNet-264 model.
This preset uses `DenseNetConfig(block_config=[6, 12, 64, 48], growth_rate=32, num_init_features=64)`.

**Total Parameters**: 33,337,704

Function Signature
------------------

.. code-block:: python

    @register_model
    def densenet_264(num_classes: int = 1000, **kwargs) -> DenseNet

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `DenseNetConfig`, excluding the preset
  `block_config`, `growth_rate`, and `num_init_features` fields.

Returns
-------

- **DenseNet**:
  A DenseNet-264 model instance with the preset
  `(block_config=[6, 12, 64, 48], growth_rate=32, num_init_features=64)`.
