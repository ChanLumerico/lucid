PVT_V2
======

.. toctree::
    :maxdepth: 1
    :hidden:

    pvt_v2_b0.rst
    pvt_v2_b1.rst
    pvt_v2_b2.rst
    pvt_v2_b2_li.rst
    pvt_v2_b3.rst
    pvt_v2_b4.rst
    pvt_v2_b5.rst

|transformer-badge| |vision-transformer-badge| 

.. autoclass:: lucid.models.PVT_V2

The `PVT_V2` class implements the second version of the Pyramid Vision Transformer (PVT-v2),
a hierarchical transformer architecture enhanced for both computational efficiency and representational power
compared to its predecessor, `PVT`.

Key Enhancements
----------------

- **Linear Attention (Optional)**: 
  PVT-v2 introduces the option to use linear attention mechanisms, 
  which reduce complexity from quadratic to linear in spatial dimensions, 
  enabling faster inference on high-resolution inputs.

- **Deeper Spatial Reduction Control**: 
  The `sr_ratios` are retained from PVT but allow finer control per stage in PVT-v2, 
  improving feature extraction and efficiency during multi-stage attention.

.. mermaid::
    :name: PVT-v2

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>pvt_v2_b0</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_OverlapPatchEmbed"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,32,56,56)</span>"];
          m3["LayerNorm"];
        end
        subgraph sg_m4["block1"]
          direction TB;
        style sg_m4 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m5["_Block_V2 x 2"]
            direction TB;
          style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m5_in(["Input"]);
            m5_out(["Output"]);
      style m5_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m5_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m6["LayerNorm"];
            m7["_LSRAttention"];
            m8["Identity"];
            m9["LayerNorm"];
            m10["_ConvMLP"];
          end
        end
        m11["LayerNorm"];
        subgraph sg_m12["_OverlapPatchEmbed"]
          direction TB;
        style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m13["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,32,56,56) → (1,64,28,28)</span>"];
          m14["LayerNorm"];
        end
        subgraph sg_m15["block2"]
          direction TB;
        style sg_m15 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m16["_Block_V2 x 2"]
            direction TB;
          style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m16_in(["Input"]);
            m16_out(["Output"]);
      style m16_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m16_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m17["LayerNorm"];
            m18["_LSRAttention"];
            m19["Identity"];
            m20["LayerNorm"];
            m21["_ConvMLP"];
          end
        end
        m22["LayerNorm"];
        subgraph sg_m23["_OverlapPatchEmbed"]
          direction TB;
        style sg_m23 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m24["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,28,28) → (1,160,14,14)</span>"];
          m25["LayerNorm"];
        end
        subgraph sg_m26["block3"]
          direction TB;
        style sg_m26 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m27["_Block_V2 x 2"]
            direction TB;
          style sg_m27 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m27_in(["Input"]);
            m27_out(["Output"]);
      style m27_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m27_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m28["LayerNorm"];
            m29["_LSRAttention"];
            m30["Identity"];
            m31["LayerNorm"];
            m32["_ConvMLP"];
          end
        end
        m33["LayerNorm"];
        subgraph sg_m34["_OverlapPatchEmbed"]
          direction TB;
        style sg_m34 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m35["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,160,14,14) → (1,256,7,7)</span>"];
          m36["LayerNorm"];
        end
        subgraph sg_m37["block4"]
          direction TB;
        style sg_m37 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m38["_Block_V2 x 2"]
            direction TB;
          style sg_m38 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m38_in(["Input"]);
            m38_out(["Output"]);
      style m38_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m38_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m39["LayerNorm"];
            m40["_LSRAttention"];
            m41["Identity"];
            m42["LayerNorm"];
            m43["_ConvMLP"];
          end
        end
        m44["LayerNorm"];
        m45["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,256) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m6 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m8 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m9 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m11 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m13 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m14 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m17 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m19 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m20 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m22 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m24 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m25 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m28 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m30 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m31 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m33 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m35 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m36 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m39 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m41 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m42 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m44 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m45 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 --> m5_out;
      m10 -.-> m8;
      m11 --> m13;
      m13 --> m14;
      m14 -.-> m17;
      m16_in -.-> m17;
      m16_out --> m22;
      m17 --> m18;
      m18 -.-> m19;
      m19 --> m16_in;
      m19 --> m20;
      m2 --> m3;
      m20 --> m21;
      m21 --> m16_out;
      m21 -.-> m19;
      m22 --> m24;
      m24 --> m25;
      m25 -.-> m28;
      m27_in -.-> m28;
      m27_out --> m33;
      m28 --> m29;
      m29 -.-> m30;
      m3 -.-> m6;
      m30 --> m27_in;
      m30 --> m31;
      m31 --> m32;
      m32 --> m27_out;
      m32 -.-> m30;
      m33 --> m35;
      m35 --> m36;
      m36 -.-> m39;
      m38_in -.-> m39;
      m38_out --> m44;
      m39 --> m40;
      m40 -.-> m41;
      m41 --> m38_in;
      m41 --> m42;
      m42 --> m43;
      m43 --> m38_out;
      m43 -.-> m41;
      m44 --> m45;
      m45 --> output;
      m5_in -.-> m6;
      m5_out --> m11;
      m6 --> m7;
      m7 -.-> m8;
      m8 --> m5_in;
      m8 --> m9;
      m9 --> m10;

Function Signature
------------------

.. code-block:: python

    class PVT_V2(nn.Module):
        def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 7,
            in_channels: int = 3,
            num_classes: int = 1000,
            embed_dims: list[int] = [64, 128, 256, 512],
            num_heads: list[int] = [1, 2, 4, 8],
            mlp_ratios: list[int] = [4, 4, 4, 4],
            qkv_bias: bool = False,
            qk_scale: float | None = None,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_layer: type[nn.Module] = nn.LayerNorm,
            depths: list[int] = [3, 4, 6, 3],
            sr_ratios: list[int] = [8, 4, 2, 1],
            num_stages: int = 4,
            linear: bool = False,
        ) -> None

Parameters
----------

- **img_size** (*int*, optional):  
  The input image size. Default is `224`.

- **patch_size** (*int*, optional):  
  The patch size for tokenization. Default is `7`.

- **in_channels** (*int*, optional):  
  Number of channels in input image. Default is `3`.

- **num_classes** (*int*, optional):  
  Number of classes for classification. Default is `1000`.

- **embed_dims** (*list[int]*, optional):  
  Embedding dimensions for each transformer stage. Default is `[64, 128, 256, 512]`.

- **num_heads** (*list[int]*, optional):  
  Number of attention heads in each stage. Default is `[1, 2, 4, 8]`.

- **mlp_ratios** (*list[int]*, optional):  
  MLP expansion ratios per stage. Default is `[4, 4, 4, 4]`.

- **qkv_bias** (*bool*, optional):  
  Whether to use bias in q/k/v projections. Default is `False`.

- **qk_scale** (*float | None*, optional):  
  Optional scaling for query-key dot product. Default is `None`.

- **drop_rate** (*float*, optional):  
  Dropout rate for MLP outputs. Default is `0.0`.

- **attn_drop_rate** (*float*, optional):  
  Dropout rate for attention weights. Default is `0.0`.

- **drop_path_rate** (*float*, optional):  
  Dropout rate for residual paths. Default is `0.0`.

- **norm_layer** (*type[nn.Module]*, optional):  
  Normalization layer to use. Default is `nn.LayerNorm`.

- **depths** (*list[int]*, optional):  
  Depth (number of blocks) at each stage. Default is `[3, 4, 6, 3]`.

- **sr_ratios** (*list[int]*, optional):  
  Spatial reduction ratios per stage. Default is `[8, 4, 2, 1]`.

- **num_stages** (*int*, optional):  
  Number of transformer stages. Default is `4`.

- **linear** (*bool*, optional):  
  Whether to use linear attention. Default is `False`.

Returns
-------
- **PVT_V2**:  
  An instance of the `PVT_V2` model for image tasks.
