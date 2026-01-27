CoAtNet
=======

.. toctree::
    :maxdepth: 1
    :hidden:

    coatnet_0.rst
    coatnet_1.rst
    coatnet_2.rst
    coatnet_3.rst
    coatnet_4.rst
    coatnet_5.rst
    coatnet_6.rst
    coatnet_7.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.CoAtNet

The `CoAtNet` module in `lucid.nn` implements the CoAtNet architecture, 
a hybrid model combining convolutional and attention-based mechanisms. 
It leverages the strengths of both convolutional neural networks (CNNs) 
and vision transformers (ViTs), making it highly efficient for image 
classification tasks. 

CoAtNet utilizes depthwise convolutions, relative position encoding, 
and pre-normalization to enhance training stability and performance.

.. mermaid::
    :name: CoAtNet
    
    %%{init: {"themeCSS":".nodeLabel, .edgeLabel, .cluster text, .node text { fill: #000000 !important; } .node foreignObject *, .cluster foreignObject * { color: #000000 !important; }"} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>coatnet_0</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05
        subgraph sg_m1["s0"]
        style sg_m1 fill:#000000,fill-opacity:0.05
          subgraph sg_m2["Sequential x 2"]
          style sg_m2 fill:#000000,fill-opacity:0.05
            m2_in(["Input"]);
            m2_out(["Output"]);
      style m2_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m2_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m3["Conv2d<br/><span style='font-size:11px;font-weight:400'>(1,3,224,224) → (1,64,112,112)</span>"];
            m4["BatchNorm2d"];
            m5["GELU"];
          end
        end
        subgraph sg_m6["s1 x 2"]
        style sg_m6 fill:#000000,fill-opacity:0.05
          m6_in(["Input"]);
          m6_out(["Output"]);
      style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m7["_MBConv"]
          style sg_m7 fill:#000000,fill-opacity:0.05
            m8["MaxPool2d<br/><span style='font-size:11px;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"];
            m9["Conv2d<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,96,56,56)</span>"];
            m10["_PreNorm<br/><span style='font-size:11px;font-weight:400'>(1,64,112,112) → (1,96,56,56)</span>"];
          end
          subgraph sg_m11["_MBConv"]
          style sg_m11 fill:#000000,fill-opacity:0.05
            m12["_PreNorm"];
          end
        end
        subgraph sg_m13["s3 x 2"]
        style sg_m13 fill:#000000,fill-opacity:0.05
          m13_in(["Input"]);
          m13_out(["Output"]);
      style m13_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m13_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m14["_Transformer"]
          style sg_m14 fill:#000000,fill-opacity:0.05
            m15(["MaxPool2d x 2<br/><span style='font-size:11px;font-weight:400'>(1,192,28,28) → (1,192,14,14)</span>"]);
            m16["Conv2d<br/><span style='font-size:11px;font-weight:400'>(1,192,14,14) → (1,384,14,14)</span>"];
            m17(["Sequential x 2<br/><span style='font-size:11px;font-weight:400'>(1,192,14,14) → (1,384,14,14)</span>"]);
          end
          subgraph sg_m18["_Transformer x 4"]
          style sg_m18 fill:#000000,fill-opacity:0.05
            m18_in(["Input"]);
            m18_out(["Output"]);
      style m18_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m18_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m19(["Sequential x 2"]);
          end
        end
        m20["AvgPool2d<br/><span style='font-size:11px;font-weight:400'>(1,768,7,7) → (1,768,1,1)</span>"];
        m21["Linear<br/><span style='font-size:11px;font-weight:400'>(1,768) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#000000;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#000000;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m3 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m4 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m5 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m8 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m9 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m15 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m16 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m20 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m21 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input -.-> m3;
      m10 --> m12;
      m12 --> m6_in;
      m12 --> m6_out;
      m13_in -.-> m15;
      m13_out --> m20;
      m15 --> m16;
      m15 --> m17;
      m16 -.-> m15;
      m17 -.-> m19;
      m18_in -.-> m19;
      m18_out --> m13_in;
      m18_out -.-> m18_in;
      m19 --> m13_out;
      m19 -.-> m18_in;
      m19 --> m18_out;
      m20 --> m21;
      m21 --> output;
      m2_in -.-> m3;
      m2_out -.-> m8;
      m3 --> m4;
      m4 --> m5;
      m5 --> m2_in;
      m5 --> m2_out;
      m6_in -.-> m8;
      m6_out -.-> m15;
      m8 --> m9;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class CoAtNet(nn.Module):
        def __init__(
            img_size: tuple[int, int],
            in_channels: int,
            num_blocks: list[int],
            channels: list[int],
            num_classes: int = 1000,
            num_heads: int = 32,
            block_types: list[str] = ["C", "C", "T", "T"],
        )

Parameters
----------

- **img_size** (*tuple[int, int]*):
  The spatial resolution of the input image (height, width).

- **in_channels** (*int*):
  The number of input channels, typically 3 for RGB images.

- **num_blocks** (*list[int]*):
  Number of blocks in each stage, defining the depth of each phase.

- **channels** (*list[int]*):
  Number of channels in each stage of the network, controlling the model width.

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **num_heads** (*int*, optional):
  Number of attention heads in the transformer-based blocks. Default is 32.

- **block_types** (*list[str]*, optional):
  Defines whether each stage uses convolution (`C`) or transformer (`T`) blocks. 
  Default is `["C", "C", "T", "T"]`.

Hybrid Architecture
-------------------

The CoAtNet model employs a hybrid structure that fuses convolutional 
and transformer blocks for enhanced representation learning:

1. **Early Convolutional Blocks**:

   - The initial stages use convolution-based feature extraction (`C` blocks).
   - These layers focus on capturing local patterns efficiently.
   - Convolutions perform feature extraction using:
     
     .. math::

         Y = W * X + b

2. **Transformer-Based Blocks**:

   - Later stages transition into transformer blocks (`T` blocks).
   - These layers incorporate self-attention to capture long-range dependencies.
   - Self-attention is computed as:
     
     .. math::

         \mathbf{A} = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V

3. **Pre-Normalization**:

   - Each transformer block applies Layer Normalization before the attention mechanism.
   - Helps improve gradient flow and stability during training.
   - The normalization step follows:
     
     .. math::

         \hat{x} = \frac{x - \mu}{\sigma + \epsilon}

4. **Relative Position Encoding**:

   - Unlike absolute position encoding in traditional ViTs, 
     CoAtNet leverages relative position encoding.
   - The attention mechanism incorporates positional information dynamically:
     
     .. math::

         A_{ij} = \frac{Q_i K_j^T}{\sqrt{d_k}} + B_{ij}

   - The relative position bias matrix \( B_{ij} \) is learnable and helps in 
     modeling spatial relationships.

5. **Depthwise Convolutions**:

   - Used to reduce computational complexity while maintaining strong feature 
     extraction capabilities.
   - Reduces the number of parameters compared to traditional convolutional layers.
   - The depthwise convolution operation is:
     
     .. math::
         Y_{i,j} = \sum_{k} X_{i+k, j+k} W_k

6. **Scaling Strategy**:

   - CoAtNet scales efficiently across depth (D), width (W), and resolution (R), 
     making it highly versatile for various image sizes and computational constraints.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Create CoAtNet with default settings
    model = models.CoAtNet(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=[2, 2, 6, 2],
        channels=[64, 128, 256, 512],
        num_classes=1000,
        num_heads=32,
        block_types=["C", "C", "T", "T"]
    )

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)
    print(output.shape)  # Shape: (1, 1000)
