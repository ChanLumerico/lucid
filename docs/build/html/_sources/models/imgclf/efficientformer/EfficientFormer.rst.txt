EfficientFormer
===============

.. toctree::
    :maxdepth: 1
    :hidden:

    efficientformer_l1.rst
    efficientformer_l3.rst
    efficientformer_l7.rst

|transformer-badge| |vision-transformer-badge| |imgclf-badge|

.. autoclass:: lucid.models.EfficientFormer

The `EfficientFormer` module implements a lightweight hybrid vision transformer 
architecture optimized for mobile and edge devices. It combines convolutional and 
transformer-based components in a streamlined hierarchical design to achieve high 
efficiency and competitive accuracy.

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
        def __init__(
            self,
            depths: list[int],
            embed_dims: int | None = None,
            in_channels: int = 3,
            num_classes: int = 1000,
            global_pool: bool = True,
            downsamples: list[bool] | None = None,
            num_vit: int = 0,
            mlp_ratios: float = 4.0,
            pool_size: int = 3,
            layer_scale_init_value: float = 1e-5,
            act_layer: type[nn.Module] = nn.GELU,
            norm_layer: type[nn.Module] = nn.BatchNorm2d,
            norm_layer_cl: type[nn.Module] = nn.LayerNorm,
            drop_rate: float = 0.0,
            proj_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
        ) -> None

Parameters
----------

- **depths** (*list[int]*):  
  Depth of each stage in the EfficientFormer hierarchy.

- **embed_dims** (*int | None*, optional):  
  Base embedding dimension. If None, it is derived from model size. Default is None.

- **in_channels** (*int*, optional):  
  Number of input image channels. Default is 3.

- **num_classes** (*int*, optional):  
  Number of classes for the classification head. Default is 1000.

- **global_pool** (*bool*, optional):  
  Whether to apply global average pooling before the classifier. Default is True.

- **downsamples** (*list[bool] | None*, optional):  
  Whether to downsample at each stage. If None, defaults to standard configuration.

- **num_vit** (*int*, optional):  
  Number of stages using transformer blocks. Remaining stages use convolutions. Default is 0.

- **mlp_ratios** (*float*, optional):  
  MLP expansion ratio in transformer blocks. Default is 4.0.

- **pool_size** (*int*, optional):  
  Kernel size for the pooling layer in convolution stages. Default is 3.

- **layer_scale_init_value** (*float*, optional):  
  Initial value for layer scale in transformer stages. Default is 1e-5.

- **act_layer** (*type[nn.Module]*, optional):  
  Activation function used throughout the model. Default is nn.GELU.

- **norm_layer** (*type[nn.Module]*, optional):  
  Normalization layer used in convolution stages. Default is nn.BatchNorm2d.

- **norm_layer_cl** (*type[nn.Module]*, optional):  
  Normalization layer used in classification layers. Default is nn.LayerNorm.

- **drop_rate** (*float*, optional):  
  Dropout rate for embedding dropout. Default is 0.0.

- **proj_drop_rate** (*float*, optional):  
  Dropout rate for projection layers. Default is 0.0.

- **drop_path_rate** (*float*, optional):  
  Drop path rate for stochastic depth. Default is 0.0.

Architecture
------------

EfficientFormer combines the strengths of convolutional inductive bias and 
transformer flexibility in a stage-wise hybrid design:

1. **Convolutional Stages**:

   - Early stages use lightweight depthwise-separable convolutions.
   - Convolution + BatchNorm + Activation (CBR) blocks dominate these layers.

2. **Transformer Stages**:

   - Later stages use efficient self-attention with MLPs and normalization.
   - Employs simplified attention to reduce complexity while maintaining accuracy.

3. **MetaBlock**:

   - The MetaBlock is the core computational unit used in both convolutional and 
     transformer stages.

   - In convolutional mode, MetaBlock uses depthwise separable convolutions with 
     residual connections.

   - In transformer mode, it applies a simplified attention mechanism followed by 
     a feed-forward MLP block.

   - It supports **layer scaling**, **drop path**, and **residual connections**, 
     enabling stable and deep training.

   - MetaBlocks unify the design across different stages, reducing architectural 
     complexity while retaining flexibility.

4. **Hierarchical Design**:

   - Embedding dimensions grow progressively through stages.
   - Optional downsampling after each stage.

5. **Classification Head**:

   - Global average pooling followed by a LayerNorm and linear classifier.

Examples
--------

**Basic Usage**

.. code-block:: python

    import lucid
    from lucid.models.transformer import EfficientFormer

    # Create a default EfficientFormer model
    model = EfficientFormer(depths=[2, 2, 6, 2], num_vit=2)

    input_tensor = lucid.randn(1, 3, 224, 224)
    output = model(input_tensor)

    print(output.shape)  # Shape: (1, 1000)

**Custom Configuration**

.. code-block:: python

    model = EfficientFormer(
        depths=[3, 3, 9, 3],
        embed_dims=64,
        num_vit=3,
        num_classes=100,
        drop_rate=0.1,
        drop_path_rate=0.1
    )

    input_tensor = lucid.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Shape: (1, 100)

.. tip::

   Increase the number of transformer stages (`num_vit`) to enhance 
   long-range feature modeling.

.. warning::

   The `depths` list must match the number of model stages, and `num_vit` 
   should not exceed that length.
