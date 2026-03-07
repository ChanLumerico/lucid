mask2former_resnet_34
=====================

.. autofunction:: lucid.models.mask2former_resnet_34

The `mask2former_resnet_34` function builds a `Mask2Former` model with a
ResNet-34 backbone preset.

**Total Parameters** (`num_labels=150`): 41,080,727

Function Signature
------------------

.. code-block:: python

    @register_model
    def mask2former_resnet_34(
        num_labels: int,
        *,
        pretrained_backbone: bool = False,
        **config_kwargs
    ) -> Mask2Former

Parameters
----------

- **num_labels** (*int*):
  Number of segmentation classes.

- **pretrained_backbone** (*bool*, optional):
  If `True`, initializes the backbone from available pretrained weights.

- **config_kwargs** (*dict*, optional):
  Additional overrides applied to `Mask2FormerConfig`.

Returns
-------

- **Mask2Former**:
  Mask2Former model configured with a ResNet-34 backbone.

Example Usage
-------------

.. code-block:: python

    from lucid.models.vision.mask2former import mask2former_resnet_34
    import lucid

    model = mask2former_resnet_34(num_labels=150)
    x = lucid.random.randn(1, 3, 224, 224)
    pred = model.predict(x)
    print(pred.shape)
