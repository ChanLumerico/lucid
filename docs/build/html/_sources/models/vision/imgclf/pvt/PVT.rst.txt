PVT
===

.. toctree::
    :maxdepth: 1
    :hidden:

    pvt_tiny.rst
    pvt_small.rst
    pvt_medium.rst
    pvt_large.rst
    pvt_huge.rst

|transformer-badge| |vision-transformer-badge| 

.. autoclass:: lucid.models.PVT

The `PVT` class implements the Pyramid Vision Transformer (PVT), a hierarchical 
vision transformer designed for image classification. PVT introduces a multi-stage 
architecture with progressive spatial reduction, enabling efficient modeling of 
global and local features. The model supports various configurations for embedding 
dimensions, attention heads, depth, and other hyperparameters.

.. mermaid::
    :name: PVT

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>pvt_medium</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_PatchEmbed x 4"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m1_in(["Input"]);
          m1_out(["Output"]);
      style m1_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m1_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,64,56,56)</span>"];
          m3["LayerNorm"];
        end
        m4(["Dropout x 4"]);
        subgraph sg_m5["block1 x 3"]
        style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m5_in(["Input"]);
          m5_out(["Output"]);
      style m5_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m5_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m6["_Block x 3"]
            direction TB;
          style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m6_in(["Input"]);
            m6_out(["Output"]);
      style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m7["LayerNorm"];
            subgraph sg_m8["_SRAttention"]
              direction TB;
            style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m9(["Linear x 2"]);
              m10(["Dropout x 3"]);
              m11["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,56,56) → (1,64,7,7)</span>"];
              m12["LayerNorm"];
            end
            m13["Identity"];
            m14["LayerNorm"];
            subgraph sg_m15["_MLP"]
              direction TB;
            style sg_m15 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m16["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,3136,64) → (1,3136,512)</span>"];
              m17["GELU"];
              m18["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,3136,512) → (1,3136,64)</span>"];
              m19["Dropout"];
            end
          end
        end
        subgraph sg_m20["block4"]
        style sg_m20 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m21["_Block x 3"]
            direction TB;
          style sg_m21 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m21_in(["Input"]);
            m21_out(["Output"]);
      style m21_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m21_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m22["LayerNorm"];
            subgraph sg_m23["_SRAttention"]
              direction TB;
            style sg_m23 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m24(["Linear x 2"]);
              m25(["Dropout x 3"]);
            end
            m26["Identity"];
            m27["LayerNorm"];
            subgraph sg_m28["_MLP"]
              direction TB;
            style sg_m28 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m29["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,50,512) → (1,50,2048)</span>"];
              m30["GELU"];
              m31["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,50,2048) → (1,50,512)</span>"];
              m32["Dropout"];
            end
          end
        end
        m33["LayerNorm"];
        m34["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,512) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m4 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m7 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m9 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m10 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m11 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m12 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m13 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m14 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m16 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m17 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m18 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m19 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m22 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m24 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m25 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m26 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m27 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m29 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m30 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m31 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m32 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m33 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m34 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input -.-> m2;
      m10 -.-> m13;
      m11 --> m12;
      m12 -.-> m9;
      m13 --> m14;
      m13 -.-> m6_in;
      m14 --> m16;
      m16 --> m17;
      m17 -.-> m19;
      m18 -.-> m19;
      m19 -.-> m13;
      m19 --> m18;
      m19 --> m5_out;
      m19 --> m6_out;
      m1_in -.-> m2;
      m1_out -.-> m4;
      m2 --> m3;
      m21_in -.-> m22;
      m21_out -.-> m21_in;
      m21_out --> m33;
      m22 --> m24;
      m24 --> m25;
      m25 -.-> m26;
      m26 -.-> m21_in;
      m26 --> m27;
      m27 --> m29;
      m29 --> m30;
      m3 --> m1_out;
      m3 -.-> m4;
      m30 -.-> m32;
      m31 -.-> m32;
      m32 --> m21_out;
      m32 -.-> m26;
      m32 --> m31;
      m33 --> m34;
      m34 --> output;
      m4 -.-> m22;
      m4 --> m5_in;
      m4 -.-> m7;
      m5_in -.-> m7;
      m5_out -.-> m1_in;
      m6_in -.-> m7;
      m6_out -.-> m1_in;
      m6_out -.-> m6_in;
      m7 -.-> m9;
      m9 --> m10;
      m9 --> m11;

Function Signature
------------------

.. code-block:: python

    class PVT(nn.Module):
        def __init__(
            self,
            img_size: int = 224,
            num_classes: int = 1000,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dims: list[int] = [64, 128, 256, 512],
            num_heads: list[int] = [1, 2, 4, 8],
            mlp_ratios: list[float] = [4.0, 4.0, 4.0, 4.0],
            qkv_bias: bool = False,
            qk_scale: float | None = None,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_layer: type[nn.Module] = nn.LayerNorm,
            depths: list[int] = [3, 4, 6, 3],
            sr_ratios: list[float] = [8.0, 4.0, 2.0, 1.0],
        ) -> None

Parameters
----------

- **img_size** (*int*, optional):  
  The input image size. Default is `224`.

- **num_classes** (*int*, optional):  
  The number of output classes for classification. Default is `1000`.

- **patch_size** (*int*, optional):  
  The size of image patches processed by the model. Default is `16`.

- **in_channels** (*int*, optional):  
  The number of input channels. Default is `3` (for RGB images).

- **embed_dims** (*list[int]*, optional):  
  A list specifying the embedding dimensions at different stages of the model.
  Default is `[64, 128, 256, 512]`.

- **num_heads** (*list[int]*, optional):  
  A list specifying the number of attention heads in each stage.
  Default is `[1, 2, 4, 8]`.

- **mlp_ratios** (*list[float]*, optional):  
  The expansion ratio for the MLP layers in each stage.
  Default is `[4.0, 4.0, 4.0, 4.0]`.

- **qkv_bias** (*bool*, optional):  
  Whether to use bias in the query, key, and value projections. Default is `False`.

- **qk_scale** (*float | None*, optional):  
  Custom scaling factor for query-key dot product attention. Default is `None`.

- **drop_rate** (*float*, optional):  
  Dropout probability for MLP layers. Default is `0.0`.

- **attn_drop_rate** (*float*, optional):  
  Dropout probability for attention weights. Default is `0.0`.

- **drop_path_rate** (*float*, optional):  
  Stochastic depth dropout rate. Default is `0.0`.

- **norm_layer** (*type[nn.Module]*, optional):  
  Normalization layer used in the transformer blocks. Default is `nn.LayerNorm`.

- **depths** (*list[int]*, optional):  
  A list specifying the number of transformer blocks in each stage.
  Default is `[3, 4, 6, 3]`.

- **sr_ratios** (*list[float]*, optional):  
  A list specifying the spatial reduction ratio for key and value projections in 
  each stage. Default is `[8.0, 4.0, 2.0, 1.0]`.

Returns
-------
- **PVT**:  
  An instance of the `PVT` class representing a Pyramid Vision Transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.PVT()
    >>> print(model)
    PVT(img_size=224, num_classes=1000, patch_size=16, embed_dims=[64, 128, 256, 512], 
        num_heads=[1, 2, 4, 8], depths=[3, 4, 6, 3], sr_ratios=[8.0, 4.0, 2.0, 1.0], ...)
