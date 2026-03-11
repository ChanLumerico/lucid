vggnet_13
=========

.. autofunction:: lucid.models.vggnet_13

The `vggnet_13` function constructs a VGGNet-13 model, 
which is a variant of the VGGNet architecture with 13 layers. It builds a
`VGGNetConfig` preset internally and forwards extra keyword arguments to that config.

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
  Additional keyword arguments passed to `VGGNetConfig`, such as
  `in_channels`, `dropout`, or `classifier_hidden_features`.

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

.. code-block:: python

    model = vggnet_13(
        num_classes=10,
        in_channels=1,
        dropout=0.25,
        classifier_hidden_features=(512, 256),
    )

    print(model.config)
