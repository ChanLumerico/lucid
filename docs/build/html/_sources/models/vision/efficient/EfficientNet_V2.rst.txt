EfficientNet_V2
===============

.. toctree::
    :maxdepth: 1
    :hidden:

    efficientnet_v2_s.rst
    efficientnet_v2_m.rst
    efficientnet_v2_l.rst
    efficientnet_v2_xl.rst

|convnet-badge| 

.. autoclass:: lucid.models.EfficientNet_V2

`EfficientNet_V2` builds on the EfficientNet architecture, which employs a compound scaling method to balance
depth, width, and resolution for optimal performance. The V2 variant introduces further improvements
such as training with larger batch sizes, using higher-resolution images, and advanced regularization techniques
like stochastic depth and progressive learning.

.. mermaid::
    :name: EfficientNetV2

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>efficientnet_v2_m</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["stem"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,24,112,112)</span>"];
          m3["BatchNorm2d"];
          m4["Swish"];
        end
        subgraph sg_m5["features"]
        style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m6["_FusedMBConv x 13"]
          style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m6_in(["Input"]);
            m6_out(["Output"]);
      style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m7["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,24,112,112) → (1,24,56,56)</span>"];
            m8["Identity"];
          end
          subgraph sg_m9["_MBConv x 44"]
          style sg_m9 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m9_in(["Input"]);
            m9_out(["Output"]);
      style m9_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m9_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m10(["Sequential x 2<br/><span style='font-size:11px;font-weight:400'>(1,80,14,14) → (1,320,7,7)</span>"]);
            m11["_SEBlock<br/><span style='font-size:11px;font-weight:400'>(1,320,7,7) → (1,320,1,1)</span>"];
            m12["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,320,7,7) → (1,160,7,7)</span>"];
          end
        end
        subgraph sg_m13["head"]
        style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m14["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,512,4,4) → (1,2048,4,4)</span>"];
          m15["BatchNorm2d"];
          m16["Swish"];
        end
        m17["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,2048,4,4) → (1,2048,1,1)</span>"];
        m18["Dropout"];
        m19["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2048) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m8 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m14 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m15 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m16 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m17 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m18 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m19 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 --> m11;
      m11 --> m12;
      m12 -.-> m9_in;
      m12 --> m9_out;
      m14 --> m15;
      m15 --> m16;
      m16 --> m17;
      m17 --> m18;
      m18 --> m19;
      m19 --> output;
      m2 --> m3;
      m3 --> m4;
      m4 -.-> m7;
      m6_in -.-> m7;
      m6_out -.-> m10;
      m6_out -.-> m6_in;
      m7 --> m8;
      m8 -.-> m6_in;
      m8 --> m6_out;
      m9_in -.-> m10;
      m9_out --> m14;
      m9_out -.-> m9_in;

Class Signature
---------------

.. code-block:: python

    class EfficientNet_V2(nn.Module):
        def __init__(
            self,
            block_cfg: list,
            num_classes: int = 1000,
            dropout: float = 0.2,
            drop_path_rate: float = 0.2,
        ) -> None

Parameters
----------

- **block_cfg** (*list*):
  A list defining the structure and parameters of the building blocks in the network. 
  Each entry specifies the configuration of a block, such as number of filters, stride, etc.

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **dropout** (*float*, optional):
  The dropout rate applied to the final fully connected layer. Default is 0.2.

- **drop_path_rate** (*float*, optional):
  The rate for stochastic depth regularization. Default is 0.2.

.. warning::

   Ensure the `block_cfg` is well-defined to avoid shape mismatches or runtime errors
   during the forward pass.
