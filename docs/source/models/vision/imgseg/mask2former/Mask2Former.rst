Mask2Former
===========
|transformer-badge| |segmentation-transformer-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    Mask2FormerConfig.rst
    mask2former_resnet_18.rst
    mask2former_resnet_34.rst
    mask2former_resnet_50.rst
    mask2former_resnet_101.rst
    mask2former_swin_tiny.rst
    mask2former_swin_small.rst
    mask2former_swin_base.rst
    mask2former_swin_large.rst

.. autoclass:: lucid.models.Mask2Former

`Mask2Former` extends mask-classification segmentation with masked attention
and multi-scale features. In this lucid implementation, it supports both
ResNet and Swin backbones through preset builders.

Class Signature
---------------

.. code-block:: python

    class Mask2Former(PreTrainedModelMixin, nn.Module):
        def __init__(
            self,
            config: Mask2FormerConfig,
            backbone: nn.Module | None = None,
        ) -> None

Parameters
----------

- **config** (*Mask2FormerConfig*):
  Model hyperparameters including backbone metadata, decoder depth, and losses.

- **backbone** (*nn.Module | None*, optional):
  Feature extractor for the pixel-level module. If `None`, a supported
  backbone can be inferred from `config.backbone_config`.

Methods
-------

.. automethod:: lucid.models.Mask2Former.forward
.. automethod:: lucid.models.Mask2Former.predict
.. automethod:: lucid.models.Mask2Former.get_auxiliary_logits
.. automethod:: lucid.models.Mask2Former.get_loss_dict
.. automethod:: lucid.models.Mask2Former.get_loss
.. automethod:: lucid.models.Mask2Former.from_pretrained

Examples
--------

**Build from Swin Preset**

.. code-block:: python

    from lucid.models.vision.mask2former import mask2former_swin_small
    import lucid

    model = mask2former_swin_small(num_labels=150)
    x = lucid.random.randn(1, 3, 224, 224)

    out = model(x)
    print(out["class_queries_logits"].shape)
    print(out["masks_queries_logits"].shape)

**Load Pretrained Lucid Weights**

.. code-block:: python

    import lucid.models as models
    import lucid.weights as W

    weight = W.Mask2Former_Swin_Small_Weights.ADE20K
    config = models.Mask2FormerConfig(**weight.config)
    model = models.Mask2Former(config).from_pretrained(weight)

**Load with Builder Shortcut**

.. code-block:: python

    import lucid.models as models
    import lucid.weights as W

    model = models.mask2former_swin_tiny(
        num_labels=150,
        weights=W.Mask2Former_Swin_Tiny_Weights.ADE20K,
    )

**Swin-Base/Large Input Resolution**

.. code-block:: python

    import lucid
    import lucid.models as models
    import lucid.weights as W

    model = models.mask2former_swin_base(
        num_labels=150,
        weights=W.Mask2Former_Swin_Base_Weights.ADE20K,
    )
    x = lucid.random.randn(1, 3, 384, 384)
    out = model(x)
    print(out["masks_queries_logits"].shape)
