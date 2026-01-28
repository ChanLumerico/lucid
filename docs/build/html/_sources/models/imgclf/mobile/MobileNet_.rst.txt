MobileNet
=========

.. toctree::
    :maxdepth: 1
    :hidden:

    mobilenet.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.MobileNet

Overview
--------

The `MobileNet` class implements the MobileNet-v1 architecture, 
which introduces depthwise separable convolutions to reduce computational 
cost while maintaining accuracy. This architecture is ideal for mobile and 
embedded vision applications.

.. mermaid::
    :name: MobileNet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>mobilenet</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["ConvBNReLU2d"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,32,112,112)</span>"];
          m3["BatchNorm2d"];
          m4["ReLU"];
        end
        subgraph sg_m5["_Depthwise"]
          direction TB;
        style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m6["depthwise x 2"]
            direction TB;
          style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m6_in(["Input"]);
            m6_out(["Output"]);
      style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m7["Conv2d"];
            m8["BatchNorm2d"];
            m9["ReLU"];
          end
        end
        subgraph sg_m10["conv3 x 3"]
          direction TB;
        style sg_m10 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m10_in(["Input"]);
          m10_out(["Output"]);
      style m10_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m10_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m11["_Depthwise x 2"]
            direction TB;
          style sg_m11 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m11_in(["Input"]);
            m11_out(["Output"]);
      style m11_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m11_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m12(["Sequential x 2<br/><span style='font-size:11px;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"]);
          end
        end
        subgraph sg_m13["_Depthwise x 2"]
          direction TB;
        style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m13_in(["Input"]);
          m13_out(["Output"]);
      style m13_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m13_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m14["depthwise x 2"]
            direction TB;
          style sg_m14 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m14_in(["Input"]);
            m14_out(["Output"]);
      style m14_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m14_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m15["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,512,14,14) → (1,512,7,7)</span>"];
            m16["BatchNorm2d"];
            m17["ReLU"];
          end
        end
        m18["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,1024,4,4) → (1,1024,1,1)</span>"];
        m19["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,1024) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m7 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m8 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m9 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m15 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m16 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m17 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m18 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m19 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10_in -.-> m12;
      m10_out -.-> m10_in;
      m10_out -.-> m15;
      m11_in -.-> m12;
      m11_out -.-> m10_in;
      m12 --> m10_out;
      m12 --> m11_in;
      m12 --> m11_out;
      m13_in -.-> m15;
      m13_out --> m18;
      m14_in -.-> m15;
      m14_out --> m13_in;
      m15 --> m16;
      m16 --> m17;
      m17 --> m13_out;
      m17 --> m14_in;
      m17 --> m14_out;
      m18 --> m19;
      m19 --> output;
      m2 --> m3;
      m3 --> m4;
      m4 -.-> m7;
      m6_in -.-> m7;
      m6_out -.-> m12;
      m7 --> m8;
      m8 --> m9;
      m9 --> m6_in;
      m9 --> m6_out;

Class Signature
---------------

.. code-block:: python

    class MobileNet(nn.Module):
        def __init__(self, width_multiplier: float, num_classes: int = 1000) -> None

Parameters
----------
- **width_multiplier** (*float*):
  Adjusts the width of the network by scaling the number of channels in each layer. 
  A higher value increases the capacity of the model, while a lower value reduces computational cost.

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Default is 1000, 
  commonly used for ImageNet.

.. tip::

  - Adjust the `width_multiplier` and `num_classes` parameters to suit specific 
    datasets or computational constraints.
