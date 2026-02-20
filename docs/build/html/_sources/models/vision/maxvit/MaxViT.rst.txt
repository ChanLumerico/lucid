MaxViT
======

.. toctree::
    :maxdepth: 1
    :hidden:

    maxvit_tiny.rst
    maxvit_small.rst
    maxvit_base.rst
    maxvit_large.rst
    maxvit_xlarge.rst

|transformer-badge| |vision-transformer-badge| 

.. autoclass:: lucid.models.MaxViT

The `MaxViT` module implements the Multi-Axis Vision Transformer architecture, 
which combines local and global attention using a novel block design 
featuring convolution, window attention, and grid attention stages.

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
        def __init__(
            in_channels: int = 3,
            depths: tuple[int, ...] = (2, 2, 5, 2),
            channels: tuple[int, ...] = (64, 128, 256, 512),
            num_classes: int = 1000,
            embed_dim: int = 64,
            num_heads: int = 32,
            grid_window_size: tuple[int, int] = (7, 7),
            attn_drop: float = 0.0,
            drop: float = 0.0,
            drop_path: float = 0.0,
            mlp_ratio: float = 4.0,
            act_layer: type[nn.Module] = nn.GELU,
            norm_layer: type[nn.Module] = nn.BatchNorm2d,
            norm_layer_tf: type[nn.Module] = nn.LayerNorm,
        )

Parameters
----------

- **in_channels** (*int*, optional):  
  Number of input image channels. Default is 3.

- **depths** (*tuple[int, ...]*, optional):  
  Number of MaxViT blocks in each stage. Default is (2, 2, 5, 2).

- **channels** (*tuple[int, ...]*, optional):  
  Number of channels in each stage. Default is (64, 128, 256, 512).

- **num_classes** (*int*, optional):  
  Number of output classes for classification. Default is 1000.

- **embed_dim** (*int*, optional):  
  Embedding dimension for the initial convolution. Default is 64.

- **num_heads** (*int*, optional):  
  Number of attention heads for both window and grid attention. Default is 32.

- **grid_window_size** (*tuple[int, int]*, optional):  
  Size of window and grid used in attention blocks. Default is (7, 7).

- **attn_drop** (*float*, optional):  
  Dropout rate for attention weights. Default is 0.0.

- **drop** (*float*, optional):  
  Dropout rate for MLP and projection outputs. Default is 0.0.

- **drop_path** (*float*, optional):  
  Drop path (stochastic depth) rate. Default is 0.0.

- **mlp_ratio** (*float*, optional):  
  Ratio of MLP hidden dimension to embedding dimension. Default is 4.0.

- **act_layer** (*type[nn.Module]*, optional):  
  Activation layer used in MLPs. Default is nn.GELU.

- **norm_layer** (*type[nn.Module]*, optional):  
  Normalization layer used in convolution stages. Default is nn.BatchNorm2d.

- **norm_layer_tf** (*type[nn.Module]*, optional):  
  Normalization layer used in transformer blocks. Default is nn.LayerNorm.

Architecture
------------

MaxViT is composed of four key stages:

1. **Convolutional Stem**:  
   - Processes the input image to obtain a patch-level feature map.

2. **MaxViT Blocks**:  
   - Each stage includes:
   
     - **MBConv Block** for local spatial learning.
     - **Window Attention** to capture fine-grained local dependencies.
     - **Grid Attention** to model global interactions across non-overlapping partitions.

3. **Stage-wise Depths and Channels**:  
   - Allows progressive increase in capacity and receptive field.

4. **Classification Head**:  
   - Global pooling followed by a linear classifier.

Mathematical Representation
---------------------------

MaxViT alternates between two types of multi-head attention:

**Window Attention:**

.. math::

    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

**Grid Attention (shifted spatial axis):**

.. math::

    \text{GridAttention}(X) = \text{Attention}(W_Q \cdot X, W_K \cdot X, W_V \cdot X)

These attention mechanisms operate in localized (window) and global (grid) spaces.

Examples
--------

**Basic Usage**

.. code-block:: python

    import lucid
    from lucid.models.transformer import MaxViT

    model = MaxViT()
    input_tensor = lucid.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # (1, 1000)

**Custom Configuration**

.. code-block:: python

    model = MaxViT(
        in_channels=3,
        depths=(2, 3, 4, 2),
        channels=(64, 128, 256, 512),
        num_classes=10,
        embed_dim=96,
        num_heads=8,
        grid_window_size=(7, 7)
    )
    input_tensor = lucid.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # (1, 10)

.. note::
   The number of stages in MaxViT is determined by the length of `depths` and `channels`.

.. tip::
   Use larger `grid_window_size` for inputs with high spatial resolution to improve 
   global attention coverage.

.. warning::
   Ensure that `num_heads` divides the embedding dimension cleanly to prevent shape 
   mismatch in attention computation.

Implementation Details
----------------------

MaxViT, introduced by Tu et al., integrates convolution and transformer mechanisms. 
It uses MBConv, window attention, and grid attention in series to model both local 
and long-range dependencies efficiently.

MaxViT's distinctive design improves training stability and performance on 
image classification tasks, particularly on large datasets with diverse object scales.

- **MBConv**: Efficient spatial feature learning with residual connections.
- **Window Attention**: Captures local patch-level context.
- **Grid Attention**: Enables long-range interaction via shifted partitions.
- **Classification Head**: Uses global average pooling followed by a linear projection.
