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

.. mermaid::
    :name: Mask2Former

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>Mask2Former</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_Mask2FormerModel"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m2["_Mask2FormerPixelLevelModule"]
            direction TB;
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m3["_Mask2FormerSwinBackbone"]
            direction TB;
            style sg_m3 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m4["Module"];
            m5["Dropout"];
            m6(["Module x 2"]);
            end
            subgraph sg_m7["_Mask2FormerPixelDecoder"]
            direction TB;
            style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8["_Mask2FormerSinePositionEmbedding"];
            m9["ModuleList"];
            m10["_Mask2FormerPixelDecoderEncoderOnly"];
            m11["Conv2d"];
            m12(["Sequential x 2"]);
            end
        end
        subgraph sg_m13["_Mask2FormerTransformerModule"]
            direction TB;
        style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m14["_Mask2FormerSinePositionEmbedding"];
            m15(["Embedding x 2"]);
            subgraph sg_m16["_Mask2FormerMaskedAttentionDecoder"]
            direction TB;
            style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m17["ModuleList"];
            m18["LayerNorm"];
            m19["_Mask2FormerMaskPredictor"];
            end
            m20["Embedding"];
        end
        end
        m21["Linear"];
        subgraph sg_m22["_Mask2FormerLoss"]
        style sg_m22 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m23["_Mask2FormerHungarianMatcher"];
        end
    end
    input["Input"];
    output["Output"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m5 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m11 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m15 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m18 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m20 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m21 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input --> m4;
    m10 --> m12;
    m11 --> m14;
    m12 --> m11;
    m14 -.-> m18;
    m17 -.-> m18;
    m18 --> m19;
    m19 --> m17;
    m19 --> m21;
    m21 --> output;
    m4 --> m5;
    m5 --> m6;
    m6 -.-> m9;
    m8 --> m10;
    m8 -.-> m9;
    m9 --> m8;

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
