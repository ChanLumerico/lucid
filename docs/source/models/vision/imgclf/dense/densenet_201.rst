densenet_201
============

.. autofunction:: lucid.models.densenet_201

The `densenet_201` function constructs a DenseNet-201 model.
This preset uses `DenseNetConfig(block_config=[6, 12, 48, 32], growth_rate=32, num_init_features=64)`.

**Total Parameters**: 20,013,928

Function Signature
------------------

.. code-block:: python

    @register_model
    def densenet_201(num_classes: int = 1000, **kwargs) -> DenseNet

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
  A DenseNet-201 model instance with the preset
  `(block_config=[6, 12, 48, 32], growth_rate=32, num_init_features=64)`.
