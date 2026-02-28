MaskFormer
==========
|transformer-badge| |segmentation-transformer-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    maskformer_resnet_18.rst
    maskformer_resnet_34.rst
    maskformer_resnet_50.rst
    maskformer_resnet_101.rst

.. autoclass:: lucid.models.MaskFormer

`MaskFormer` reformulates segmentation as a mask classification problem.
The model combines a CNN backbone + pixel decoder with a Transformer decoder
that predicts query-level class logits and mask embeddings.

Class Signature
---------------

.. code-block:: python

    class MaskFormer(PreTrainedModelMixin, nn.Module):
        def __init__(
            self,
            config: MaskFormerConfig,
            backbone: nn.Module | None = None,
        ) -> None

Parameters
----------

- **config** (*MaskFormerConfig*):
  Model hyperparameters including label count, decoder depth/width, and
  backbone metadata.

- **backbone** (*nn.Module | None*, optional):
  Feature extractor used by the pixel-level module. If `None`, a backbone may
  be inferred from `config.backbone_config` when supported.

Configuration
-------------

.. autoclass:: lucid.models.MaskFormerConfig

Methods
-------

.. automethod:: lucid.models.MaskFormer.forward
.. automethod:: lucid.models.MaskFormer.predict
.. automethod:: lucid.models.MaskFormer.get_logits
.. automethod:: lucid.models.MaskFormer.get_loss_dict
.. automethod:: lucid.models.MaskFormer.get_loss

Examples
--------

**Build from Preset Builder**

.. code-block:: python

    from lucid.models.vision.maskformer import maskformer_resnet_50
    import lucid

    model = maskformer_resnet_50(num_labels=150)
    x = lucid.random.randn(1, 3, 512, 512)

    out = model(x)
    print(out["class_queries_logits"].shape)
    print(out["masks_queries_logits"].shape)

**Load Pretrained ADE20K Weights**

.. code-block:: python

    import lucid.models as models
    import lucid.weights as W

    weight = W.MaskFormer_ResNet_50_Weights.ADE20K
    config = models.MaskFormerConfig(**weight.config)
    model = models.MaskFormer(config).from_pretrained(weight)

    print(model.parameter_size)
