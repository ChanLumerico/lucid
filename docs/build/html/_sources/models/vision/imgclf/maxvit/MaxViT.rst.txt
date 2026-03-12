MaxViT
======

.. toctree::
    :maxdepth: 1
    :hidden:

    MaxViTConfig.rst
    maxvit_tiny.rst
    maxvit_small.rst
    maxvit_base.rst
    maxvit_large.rst
    maxvit_xlarge.rst

|transformer-badge| |vision-transformer-badge|

.. autoclass:: lucid.models.MaxViT

The `MaxViT` module implements the Multi-Axis Vision Transformer architecture,
combining MBConv blocks, window attention, and grid attention in a hierarchical
image backbone. Model structure is defined through `MaxViTConfig`.

.. mermaid::
    :name: MaxViT

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>maxvit_base</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["stem"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,64,112,112)</span>"];
          m3["GELU"];
          m4["Conv2d"];
          m5["GELU"];
        end
        subgraph sg_m6["stages"]
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m7["_MaxViTStage x 4"]
          style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m7_in(["Input"]);
            m7_out(["Output"]);
      style m7_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m7_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            subgraph sg_m8["blocks"]
              direction TB;
            style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              subgraph sg_m9["_MaxViTBlock x 2"]
                direction TB;
              style sg_m9 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
                m9_in(["Input"]);
                m9_out(["Output"]);
      style m9_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m9_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
                m10["_MBConv<br/><span style='font-size:11px;font-weight:400'>(1,64,112,112) → (1,96,56,56)</span>"];
                m11(["_MaxViTTransformerBlock x 2"]);
              end
            end
          end
        end
        m12["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,768) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m4 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m5 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m12 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 --> m11;
      m11 --> m7_out;
      m11 --> m9_in;
      m11 --> m9_out;
      m12 --> output;
      m2 --> m3;
      m3 --> m4;
      m4 --> m5;
      m5 -.-> m10;
      m7_in -.-> m10;
      m7_out --> m12;
      m7_out -.-> m7_in;
      m9_in -.-> m10;
      m9_out -.-> m7_in;

Class Signature
---------------

.. code-block:: python

    class MaxViT(nn.Module):
        def __init__(self, config: MaxViTConfig) -> None

Parameters
----------

- **config** (*MaxViTConfig*):
  Configuration object describing the stem width, stage depths, stage channels,
  attention heads, window size, dropout settings, and classifier size.

Architecture
------------

MaxViT is composed of four key stages:

1. **Convolutional Stem**:

   - A two-layer convolutional stem converts the input image to a feature map.

2. **MBConv + Transformer Blocks**:

   - Each MaxViT block starts with an MBConv path for local spatial modeling.
   - A window-attention block captures local token interactions.
   - A grid-attention block captures longer-range interactions.

3. **Hierarchical Stages**:

   - Later stages widen the channel count while reducing spatial resolution.
   - Depth and width are controlled stage-by-stage through the config.

4. **Classification Head**:

   - Global average pooling is applied over the final feature map.
   - A linear head produces the class logits.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.MaxViTConfig(
    ...     in_channels=3,
    ...     depths=(2, 2, 5, 2),
    ...     channels=(64, 128, 256, 512),
    ...     num_classes=1000,
    ...     embed_dim=64,
    ... )
    >>> model = models.MaxViT(config)
