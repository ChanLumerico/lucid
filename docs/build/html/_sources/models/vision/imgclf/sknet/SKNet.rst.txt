SKNet
=====

.. toctree::
    :maxdepth: 1
    :hidden:

    sk_resnet_18.rst
    sk_resnet_34.rst
    sk_resnet_50.rst
    
    sk_resnext_50_32x4d.rst

|convnet-badge| 

.. autoclass:: lucid.models.SKNet

The `SKNet` class extends the `ResNet` architecture by incorporating Selective Kernel (SK) blocks,
which dynamically adjust receptive fields via attention mechanisms. This enables the network to
adaptively fuse multi-scale features, improving performance on tasks involving objects of varying scales.

.. mermaid::
    :name: SKNet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>sk_resnet_50</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["stem"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,64,112,112)</span>"];
          m3["BatchNorm2d"];
          m4["ReLU"];
        end
        m5["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"];
        subgraph sg_m6["layer1 x 4"]
          direction TB;
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m6_in(["Input"]);
          m6_out(["Output"]);
      style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m7["_SKResNetBottleneck"]
            direction TB;
          style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8["ConvBNReLU2d"];
            m9["SelectiveKernel"];
            m10["BatchNorm2d"];
            m11["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"];
            m12["ReLU"];
            m13["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"];
          end
          subgraph sg_m14["_SKResNetBottleneck x 2"]
            direction TB;
          style sg_m14 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m14_in(["Input"]);
            m14_out(["Output"]);
      style m14_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m14_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m15["ConvBNReLU2d<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56) → (1,64,56,56)</span>"];
            m16["SelectiveKernel"];
            m17["BatchNorm2d"];
            m18["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"];
            m19["ReLU"];
          end
        end
        m20["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,2048,7,7) → (1,2048,1,1)</span>"];
        m21["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2048) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m5 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m10 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m12 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m17 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m19 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m20 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m21 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 -.-> m12;
      m11 --> m13;
      m12 --> m11;
      m12 -.-> m15;
      m13 -.-> m12;
      m14_in -.-> m15;
      m14_out -.-> m6_in;
      m15 --> m16;
      m16 --> m17;
      m17 -.-> m19;
      m18 -.-> m19;
      m19 --> m14_in;
      m19 --> m14_out;
      m19 --> m18;
      m19 --> m6_out;
      m2 --> m3;
      m20 --> m21;
      m21 --> output;
      m3 --> m4;
      m4 --> m5;
      m5 -.-> m8;
      m6_in -.-> m8;
      m6_out --> m20;
      m6_out -.-> m6_in;
      m8 --> m9;
      m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class lucid.nn.SKNet(
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        kernel_sizes: list[int] = [3, 5],
        base_width: int = 64,
        cardinality: int = 1,
    )

Parameters
----------
- **block** (*nn.Module*):
  The building block module used for the SKNet layers. Typically an SKBlock or compatible 
  block type.

- **layers** (*list[int]*):
  Specifies the number of blocks in each stage of the network.

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default: 1000.

- **kernel_sizes** (*list[int]*, optional):
  Specifies the sizes of kernels to be used in the SK blocks for multi-scale processing. 
  Default: [3, 5].

- **base_width** (*int*, optional):
  Base width of the feature maps in the SK blocks. Default: 64.

- **cardinality** (*int*, optional):
  The number of parallel convolutional groups (grouped convolutions) in the SK blocks. 
  Default: 1.

Attributes
----------
- **kernel_sizes** (*list[int]*):
  Stores the kernel sizes used in the SK blocks.

- **base_width** (*int*):
  Stores the base width of feature maps.

- **cardinality** (*int*):
  Stores the number of groups for grouped convolutions.

- **layers** (*list[nn.Module]*):
  A list of stages, each containing a sequence of SK blocks.

Forward Calculation
--------------------
The forward pass of the `SKNet` model includes:

1. **Stem**: Initial convolutional layers for feature extraction.
2. **Selective Kernel Stages**: Each stage applies a series of SK blocks configured via `layers`.
3. **Global Pooling**: A global average pooling layer reduces spatial dimensions.
4. **Classifier**: A fully connected layer maps the features to class scores.

.. math::

    \text{output} = \text{FC}(\text{GAP}(\text{SKBlocks}(\text{Stem}(\text{input}))))

.. note::

   - The `SKNet` is well-suited for tasks requiring multi-scale feature representation.
   - Increasing the `kernel_sizes` parameter allows the model to capture features over 
     larger receptive fields.
