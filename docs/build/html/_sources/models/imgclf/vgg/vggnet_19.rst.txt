vggnet_19
=========

.. autofunction:: lucid.models.vggnet_19

The `vggnet_19` function constructs a VGGNet-19 model, 
which is a variant of the VGGNet architecture with 19 layers.

**Total Parameters**: 143,667,240

Function Signature
------------------

.. code-block:: python

    @register_model
    def vggnet_19(num_classes: int = 1000, **kwargs) -> VGGNet

Parameters
----------

- **num_classes** (*int*, optional):
  The number of classes for the classification output. Defaults to 1000.

- **kwargs**:
  Additional keyword arguments passed to the `VGGNet` constructor.

Returns
-------

- **VGGNet**:
  A VGGNet-19 model initialized with the specified parameters.

Examples
--------

.. code-block:: python

    from lucid.models import vggnet_19

    # Create a VGGNet-19 model
    model = vggnet_19(num_classes=100)

    print(model)
