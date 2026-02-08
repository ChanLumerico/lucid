SwinTransformer
===============

.. toctree::
    :maxdepth: 1
    :hidden:

    swin_tiny.rst
    swin_small.rst
    swin_base.rst
    swin_large.rst

|transformer-badge| |vision-transformer-badge| |imgclf-badge|

.. autoclass:: lucid.models.SwinTransformer

The `SwinTransformer` class implements a hierarchical vision transformer with shifted
windows, designed for image recognition and dense prediction tasks. Unlike the original
Vision Transformer (ViT), which processes fixed-size patches in a flat manner, the Swin
Transformer divides the image into patches, computes self-attention within local windows,
and then shifts these windows to enable cross-window interactions. This design improves
computational efficiency and allows the model to capture both local and global dependencies.

.. mermaid::
    :name: SwinTransformer

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>swin_base</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_PatchEmbed"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,128,56,56)</span>"];
          m3["LayerNorm"];
        end
        m4["Dropout"];
        subgraph sg_m5["layers"]
        style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m6["_BasicLayer x 3"]
            direction TB;
          style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m6_in(["Input"]);
            m6_out(["Output"]);
      style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            subgraph sg_m7["blocks"]
              direction TB;
            style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m8(["_SwinTransformerBlock x 2"]);
            end
            subgraph sg_m9["_PatchMerging"]
              direction TB;
            style sg_m9 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m10["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,784,512) → (1,784,256)</span>"];
              m11["LayerNorm"];
            end
          end
          subgraph sg_m12["_BasicLayer"]
            direction TB;
          style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m13["blocks"]
              direction TB;
            style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m14(["_SwinTransformerBlock x 2"]);
            end
          end
        end
        m15["LayerNorm"];
        m16["AdaptiveAvgPool1d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,1024,49) → (1,1024,1)</span>"];
        m17["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,1024) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m4 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m10 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m11 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m15 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m16 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m17 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 -.-> m6_in;
      m11 --> m10;
      m11 --> m6_out;
      m14 --> m15;
      m15 --> m16;
      m16 --> m17;
      m17 --> output;
      m2 --> m3;
      m3 --> m4;
      m4 -.-> m8;
      m6_in -.-> m8;
      m6_out --> m14;
      m6_out -.-> m6_in;
      m8 --> m11;

Class Signature
---------------

.. code-block:: python

    class SwinTransformer(
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        windows_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        abs_pos_emb: bool = False,
        patch_norm: bool = True,
    )

Parameters
----------
- **img_size** (*int*):
  Size of the input image (assumes square images).

- **patch_size** (*int*):
  Size of the patches the image is divided into.

- **in_channels** (*int*):
  Number of input channels (e.g., 3 for RGB images).

- **num_classes** (*int*):
  Number of output classes for classification.

- **embed_dim** (*int*):
  Dimension of the embedding for the first stage.

- **depths** (*list[int]*):
  A list specifying the number of transformer blocks in each stage.

- **num_heads** (*list[int]*):
  A list specifying the number of attention heads in each stage.

- **windows_size** (*int*):
  Size of the local window for self-attention.

- **mlp_ratio** (*float*):
  Ratio of the hidden dimension in the MLP relative to the embedding dimension.

- **qkv_bias** (*bool*):
  Whether to include a learnable bias in the query, key, and value projections.

- **qk_scale** (*float | None*):
  Override the default scaling for the query and key, if provided.

- **drop_rate** (*float*):
  Dropout probability applied throughout the model.

- **attn_drop_rate** (*float*):
  Dropout probability for the attention weights.

- **drop_path_rate** (*float*):
  Stochastic depth rate for regularization.

- **norm_layer** (*Type[nn.Module]*):
  Normalization layer to be used (default is `nn.LayerNorm`).

- **abs_pos_emb** (*bool*):
  Whether to use absolute positional embedding.

- **patch_norm** (*bool*):
  Whether to apply normalization after patch embedding.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> swin = models.SwinTransformer(
    ...     img_size=224,
    ...     patch_size=4,
    ...     in_channels=3,
    ...     num_classes=1000,
    ...     embed_dim=96,
    ...     depths=[2, 2, 6, 2],
    ...     num_heads=[3, 6, 12, 24],
    ... )
