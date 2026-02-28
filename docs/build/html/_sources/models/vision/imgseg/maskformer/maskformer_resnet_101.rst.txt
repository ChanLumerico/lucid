maskformer_resnet_101
=====================

.. autofunction:: lucid.models.maskformer_resnet_101

The `maskformer_resnet_101` function builds a `MaskFormer` model with a
ResNet-101 backbone preset.

**Total Parameters** (`num_labels=150`): 60,299,991

Function Signature
------------------

.. code-block:: python

    @register_model
    def maskformer_resnet_101(
        num_labels: int,
        *,
        pretrained_backbone: bool = False,
        **config_kwargs
    ) -> MaskFormer

Parameters
----------

- **num_labels** (*int*):
  Number of semantic classes for segmentation.

- **pretrained_backbone** (*bool*, optional):
  If `True`, initializes the ResNet-101 backbone with pretrained
  classification weights.

- **config_kwargs** (*dict*, optional):
  Additional overrides applied to `MaskFormerConfig`.

Returns
-------

- **MaskFormer**:
  MaskFormer model configured with a ResNet-101 backbone.

Example Usage
-------------

.. code-block:: python

    from lucid.models.vision.maskformer import maskformer_resnet_101
    import lucid

    model = maskformer_resnet_101(num_labels=150)
    x = lucid.random.randn(1, 3, 512, 512)
    pred = model.predict(x)
    print(pred.shape)
