densenet_121
============

.. autofunction:: lucid.models.densenet_121

The `densenet_121` function constructs a DenseNet-121 model.
This preset uses `DenseNetConfig(block_config=[6, 12, 24, 16], growth_rate=32, num_init_features=64)`.

**Total Parameters**: 7,978,856

Function Signature
------------------

.. code-block:: python

    @register_model
    def densenet_121(num_classes: int = 1000, **kwargs) -> DenseNet

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
  A DenseNet-121 model instance with the preset
  `(block_config=[6, 12, 24, 16], growth_rate=32, num_init_features=64)`.
