fcn_resnet_50
=============

.. autofunction:: lucid.models.fcn_resnet_50

The `fcn_resnet_50` function builds an `FCN` model with a ResNet-50 backbone
preset for semantic segmentation.

**Total Parameters** (`num_classes=21`): 35,322,218

Function Signature
------------------

.. code-block:: python

    @register_model
    def fcn_resnet_50(
        num_classes: int = 21,
        in_channels: int = 3,
        aux_loss: bool = True,
        **kwargs
    ) -> FCN

Parameters
----------

- **num_classes** (*int*): Number of segmentation classes.
- **in_channels** (*int*): Number of input channels.
- **aux_loss** (*bool*): If `True`, enables the auxiliary classifier head.
- **kwargs** (*dict*, optional): Additional overrides applied to `FCNConfig`.

Returns
-------

- **FCN**:
  FCN model configured with a ResNet-50 backbone.

Example Usage
-------------

.. code-block:: python

    from lucid.models import fcn_resnet_50
    import lucid

    model = fcn_resnet_50(num_classes=21)
    x = lucid.random.randn(1, 3, 224, 224)
    logits = model(x)
    print(logits.shape)
