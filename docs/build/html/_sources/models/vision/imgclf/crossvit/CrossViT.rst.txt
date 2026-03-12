CrossViT
========

.. toctree::
    :maxdepth: 1
    :hidden:

    CrossViTConfig.rst
    crossvit_tiny.rst
    crossvit_small.rst
    crossvit_base.rst
    crossvit_9.rst
    crossvit_15.rst
    crossvit_18.rst
    crossvit_9_dagger.rst
    crossvit_15_dagger.rst
    crossvit_18_dagger.rst

|transformer-badge| |vision-transformer-badge|

.. autoclass:: lucid.models.CrossViT

The `CrossViT` module implements the Cross-Attention Vision Transformer
architecture, which combines multiple transformer branches with cross-attention
fusion to exchange information across scales. Model structure is defined
through `CrossViTConfig`.

.. mermaid::
    :name: CrossViT

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>crossvit_base</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m1(["ParameterList x 2"]);
        subgraph sg_m2["patch_embed"]
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m3["_PatchEmbed x 2"]
          style sg_m3 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3_in(["Input"]);
            m3_out(["Output"]);
      style m3_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m3_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m4["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,240,240) → (1,384,20,20)</span>"];
          end
        end
        m5["Dropout"];
        subgraph sg_m6["blocks"]
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m7["_MultiScaleBlock x 3"]
          style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m7_in(["Input"]);
            m7_out(["Output"]);
      style m7_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m7_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            subgraph sg_m8["blocks x 2"]
            style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m8_in(["Input"]);
              m8_out(["Output"]);
      style m8_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m8_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
              m9(["Sequential x 2"]);
            end
            subgraph sg_m10["fusion"]
            style sg_m10 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m11(["_CrossAttentionBlock x 2<br/><span style='font-size:11px;font-weight:400'>(1,197,768) → (1,1,768)</span>"]);
            end
            subgraph sg_m12["revert_projs"]
            style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m13(["Sequential x 2<br/><span style='font-size:11px;font-weight:400'>(1,1,768) → (1,1,384)</span>"]);
            end
          end
        end
        subgraph sg_m14["norm"]
        style sg_m14 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m15(["LayerNorm x 2"]);
        end
        subgraph sg_m16["head"]
        style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m17(["Linear x 2<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,384) → (1,1000)</span>"]);
        end
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m4 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m5 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m15 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m17 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input -.-> m4;
      m11 --> m13;
      m13 -.-> m11;
      m13 -.-> m7_in;
      m13 --> m7_out;
      m15 --> m17;
      m17 --> output;
      m3_in -.-> m4;
      m3_out -.-> m5;
      m4 --> m3_out;
      m4 -.-> m5;
      m5 --> m3_in;
      m5 -.-> m9;
      m7_in -.-> m9;
      m7_out --> m15;
      m7_out -.-> m7_in;
      m8_in -.-> m9;
      m8_out -.-> m11;
      m9 --> m8_in;
      m9 --> m8_out;

Class Signature
---------------

.. code-block:: python

    class CrossViT(nn.Module):
        def __init__(self, config: CrossViTConfig) -> None

Parameters
----------

- **config** (*CrossViTConfig*):
  Configuration object describing the per-branch image sizes, patch sizes,
  embedding widths, multi-scale block schedule, attention heads, dropout
  settings, and optional dagger-style multi-convolution patch embedding.

Architecture
------------

The CrossViT architecture consists of:

1. **Multi-scale Patch Embedding**:

   - Different patch sizes capture complementary fine-grained and coarse image
     features.
   - Each branch keeps its own class token and positional embedding.

2. **Branch-local Transformer Blocks**:

   - Self-attention layers process each branch independently.
   - Branch depth is controlled stage-by-stage through the config.

3. **Cross-attention Fusion**:

   - Cross-attention exchanges class-token information between branches.
   - Fusion blocks project across embedding widths and then project back.

4. **Classification Head**:

   - Each branch emits a class-token logit.
   - The final prediction is the mean of per-branch logits.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.CrossViTConfig(
    ...     img_size=(240, 224),
    ...     patch_size=(12, 16),
    ...     num_classes=1000,
    ...     embed_dim=(192, 384),
    ...     depth=((1, 3, 1), (1, 3, 1), (1, 3, 1)),
    ...     num_heads=(6, 12),
    ...     mlp_ratio=(2.0, 2.0, 4.0),
    ... )
    >>> model = models.CrossViT(config)
