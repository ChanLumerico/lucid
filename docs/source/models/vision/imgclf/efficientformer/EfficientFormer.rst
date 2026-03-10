EfficientFormer
===============

.. toctree::
    :maxdepth: 1
    :hidden:

    EfficientFormerConfig.rst
    efficientformer_l1.rst
    efficientformer_l3.rst
    efficientformer_l7.rst

|transformer-badge| |vision-transformer-badge|

.. autoclass:: lucid.models.EfficientFormer

The `EfficientFormer` module implements a lightweight hybrid vision
architecture that combines convolutional meta-blocks and transformer-style
token blocks in a hierarchical pipeline. Model structure is defined through
`EfficientFormerConfig`.

.. mermaid::
    :name: EfficientFormer

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>efficientformer_l1</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["stem"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,24,112,112)</span>"];
          m3["BatchNorm2d"];
          m4["ReLU"];
          m5["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,24,112,112) → (1,48,56,56)</span>"];
          m6["BatchNorm2d"];
          m7["ReLU"];
        end
        subgraph sg_m8["stages"]
        style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m9["_EfficientFormerStage"]
          style sg_m9 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m10["Identity"];
            m11["Sequential"];
          end
          subgraph sg_m12["_EfficientFormerStage x 3"]
          style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m12_in(["Input"]);
            m12_out(["Output"]);
      style m12_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m12_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m13["_Downsample<br/><span style='font-size:11px;font-weight:400'>(1,48,56,56) → (1,96,28,28)</span>"];
            m14["Sequential"];
          end
        end
        m15["LayerNorm"];
        m16["Dropout"];
        m17["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,448) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m5 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m6 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m7 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m10 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m15 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m16 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m17 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 --> m11;
      m11 -.-> m13;
      m12_in -.-> m13;
      m12_out -.-> m12_in;
      m12_out --> m15;
      m13 --> m14;
      m14 -.-> m12_in;
      m14 --> m12_out;
      m15 --> m16;
      m16 --> m17;
      m17 --> output;
      m2 --> m3;
      m3 --> m4;
      m4 --> m5;
      m5 --> m6;
      m6 --> m7;
      m7 --> m10;

Class Signature
---------------

.. code-block:: python

    class EfficientFormer(nn.Module):
        def __init__(self, config: EfficientFormerConfig) -> None

Parameters
----------

- **config** (*EfficientFormerConfig*):
  Configuration object describing the stage depths, stage widths,
  downsampling schedule, number of final-stage token blocks, and classifier
  settings.

Architecture
------------

EfficientFormer uses a stage-wise hybrid design:

1. **Convolutional Stem**:

   - Two strided convolutions convert the image into a stage-1 feature map.

2. **MetaBlock Stages**:

   - Early stages use 2D pooling-based meta-blocks with convolutional MLPs.
   - The final stage can switch the last `num_vit` blocks to token-space
     attention blocks.

3. **Hierarchical Downsampling**:

   - Stage widths increase progressively through the network.
   - Downsampling is configurable per stage.

4. **Classification Head**:

   - The final token or pooled sequence is normalized and projected to
     `num_classes`.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.EfficientFormerConfig(
    ...     depths=(3, 2, 6, 4),
    ...     embed_dims=(48, 96, 224, 448),
    ...     num_classes=1000,
    ...     num_vit=1,
    ... )
    >>> model = models.EfficientFormer(config)
