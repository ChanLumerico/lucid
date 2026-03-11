mobilenet
=========

.. autofunction:: lucid.models.mobilenet

The `mobilenet` function constructs the MobileNet-v1 model.

**Total Parameters**: 4,232,008

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet(width_multiplier: float = 1.0, num_classes: int = 1000, **kwargs) -> MobileNet

Parameters
----------

- **width_multiplier** (*float*, optional):
  Width scaling factor applied to the MobileNet-v1 channels. Default is `1.0`.
- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `MobileNetConfig`, excluding `width_multiplier`.

Returns
-------

- **MobileNet**:
  A MobileNet-v1 model instance constructed from `MobileNetConfig`.
