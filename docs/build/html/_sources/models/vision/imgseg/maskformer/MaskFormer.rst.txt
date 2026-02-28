MaskFormer
==========
|transformer-badge| |segmentation-transformer-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    MaskFormerConfig.rst
    maskformer_resnet_18.rst
    maskformer_resnet_34.rst
    maskformer_resnet_50.rst
    maskformer_resnet_101.rst

.. autoclass:: lucid.models.MaskFormer

`MaskFormer` reformulates segmentation as a mask classification problem.
The model combines a CNN backbone + pixel decoder with a Transformer decoder
that predicts query-level class logits and mask embeddings.

.. mermaid::
    :name: MaskFormer

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>MaskFormer</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_MaskFormerModel"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m2["_MaskFormerPixelLevelModule"]
            direction TB;
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m3["_MaskFormerResNetBackbone"]
            direction TB;
            style sg_m3 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m4["Module"];
            m5["MaxPool2d"];
            m6["Module"];
            end
            subgraph sg_m7["_MaskFormerPixelDecoder"]
            direction TB;
            style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8["_MaskFormerFPNModel"];
            m9["Conv2d"];
            end
        end
        subgraph sg_m10["_MaskFormerTransformerModule"]
            direction TB;
        style sg_m10 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m11["_MaskFormerSinePositionEmbedding"]
            direction TB;
            style sg_m11 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m12["SinusoidalPosEmbedding"];
            end
            m13["Embedding"];
            m14["Conv2d"];
            subgraph sg_m15["_DETRDecoder"]
            direction TB;
            style sg_m15 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m16["ModuleList"];
            m17["LayerNorm"];
            end
        end
        end
        m18["Linear"];
        subgraph sg_m19["_MaskFormerMLPPredictionHead"]
        style sg_m19 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m20["_PredictionBlock x 2"]
            direction TB;
        style sg_m20 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m20_in(["Input"]);
            m20_out(["Output"]);
    style m20_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m20_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m21["Linear"];
            m22["ReLU"];
        end
        subgraph sg_m23["_PredictionBlock"]
            direction TB;
        style sg_m23 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m24["Linear"];
            m25["Identity"];
        end
        end
        m28["_MaskFormerHungarianMatcher"];
        subgraph sg_m27["_MaskFormerLoss"]
        style sg_m27 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m28["_MaskFormerHungarianMatcher"];
        end
    end
    input["Input"];
    output["Output"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m5 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
    style m9 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m12 fill:#e2e8f0,stroke:#334155,stroke-width:1px;
    style m13 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m14 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m17 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m18 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m21 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m22 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m24 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m25 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input --> m4;
    m14 --> m16;
    m16 --> m17;
    m17 --> m18;
    m18 -.-> m21;
    m20_in -.-> m21;
    m20_out --> m24;
    m21 --> m22;
    m22 --> m20_in;
    m22 --> m20_out;
    m24 --> m25;
    m25 --> output;
    m4 --> m5;
    m5 --> m6;
    m6 --> m8;
    m8 --> m9;
    m9 --> m14;

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
