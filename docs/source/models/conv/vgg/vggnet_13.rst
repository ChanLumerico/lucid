models.vggnet_13
================

.. autofunction:: lucid.models.vggnet_13

The `vggnet_13` function constructs a VGGNet-13 model, 
which is a variant of the VGGNet architecture with 13 layers.

**Total Parameters**: 133,047,848

Function Signature
------------------

.. code-block:: python

    @register_model
    def vggnet_13(num_classes: int = 1000, **kwargs) -> VGGNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of classes for the classification output. Defaults to 1000.

- **kwargs**:
  Additional keyword arguments passed to the `VGGNet` constructor.

Returns
-------

- **VGGNet**:
  A VGGNet-13 model initialized with the specified parameters.

Examples
--------

.. code-block:: python

    from lucid.models import vggnet_13

    # Create a VGGNet-13 model
    model = vggnet_13(num_classes=100)

    print(model)
