models.vggnet_16
================

.. autofunction:: lucid.models.vggnet_16

The `vggnet_16` function constructs a VGGNet-16 model, 
which is a variant of the VGGNet architecture with 16 layers.

**Total Parameters**: 138,357,544

Function Signature
------------------

.. code-block:: python

    def vggnet_16(num_classes: int = 1000, **kwargs) -> VGGNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of classes for the classification output. Defaults to 1000.

- **kwargs**:
  Additional keyword arguments passed to the `VGGNet` constructor.

Returns
-------

- **VGGNet**:
  A VGGNet-16 model initialized with the specified parameters.

Examples
--------

.. code-block:: python

    from lucid.models import vggnet_16

    # Create a VGGNet-16 model
    model = vggnet_16(num_classes=100)

    print(model)
