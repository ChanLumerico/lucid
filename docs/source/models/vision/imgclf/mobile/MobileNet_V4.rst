MobileNet_V4
============

.. toctree::
    :maxdepth: 1
    :hidden:

    mobilenet_v4_conv_small.rst
    mobilenet_v4_conv_medium.rst
    mobilenet_v4_conv_large.rst
    mobilenet_v4_hybrid_medium.rst
    mobilenet_v4_hybrid_large.rst

|convnet-badge| 

.. autoclass:: lucid.models.MobileNet_V4

Overview
--------
The `MobileNet_V4` class provides the foundational architecture for the MobileNet-v4 model family.
Building upon earlier MobileNet designs, it emphasizes both efficiency and flexibility, making it ideal
for mobile and embedded applications. By utilizing a configurable design through a dictionary-based
parameterization, this base class allows developers to easily experiment with different architectural
variants, striking a balance between computational cost and model performance.

.. mermaid::
    :name: MobileNet-v4

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>mobilenet_v4_conv_medium</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["conv0"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m2["convbn_0"]
            direction TB;
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,32,112,112)</span>"];
            m4["BatchNorm2d"];
            m5["ReLU6"];
        end
        end
        subgraph sg_m6["layers"]
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m7["Sequential"]
            direction TB;
        style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m8["_InvertedResidual"]
            direction TB;
            style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m9["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,32,112,112) → (1,48,56,56)</span>"];
            end
        end
        subgraph sg_m10["Sequential x 3"]
            direction TB;
        style sg_m10 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m10_in(["Input"]);
            m10_out(["Output"]);
    style m10_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m10_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            subgraph sg_m11["_UniversalInvertedBottleneck x 2"]
            direction TB;
            style sg_m11 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m11_in(["Input"]);
            m11_out(["Output"]);
    style m11_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m11_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m12(["Sequential x 4"]);
            end
        end
        subgraph sg_m13["Sequential"]
            direction TB;
        style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m14["convbn_0 x 2"]
            direction TB;
            style sg_m14 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m14_in(["Input"]);
            m14_out(["Output"]);
    style m14_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m14_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m15["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,256,7,7) → (1,960,7,7)</span>"];
            m16["BatchNorm2d"];
            m17["ReLU6"];
            end
        end
        end
        m18["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,1280,7,7) → (1,1280,1,1)</span>"];
        m19["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,1280) → (1,1000)</span>"];
    end
    input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
    output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m3 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m4 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m5 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m15 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m16 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m17 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m18 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
    style m19 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input --> m3;
    m10_in -.-> m12;
    m10_out -.-> m10_in;
    m10_out -.-> m15;
    m11_in -.-> m12;
    m11_out -.-> m10_in;
    m12 --> m10_out;
    m12 --> m11_in;
    m12 --> m11_out;
    m14_in -.-> m15;
    m14_out --> m18;
    m15 --> m16;
    m16 --> m17;
    m17 --> m14_in;
    m17 --> m14_out;
    m18 --> m19;
    m19 --> output;
    m3 --> m4;
    m4 --> m5;
    m5 --> m9;
    m9 -.-> m12;

Class Signature
---------------
.. code-block:: python

    class MobileNet_V4(nn.Module):
        def __init__(self, cfg: Dict[str, dict], num_classes: int = 1000) -> None

Parameters
----------
- **cfg** (*Dict[str, dict]*):  
  A dictionary that holds the configuration for various network blocks. Each key in the dictionary
  represents a distinct module or layer group, and its corresponding value is another dictionary 
  that specifies parameters (such as kernel size, expansion factors, filter counts, strides, etc.) 
  for that module.

- **num_classes** (*int*, optional):  
  Specifies the number of output classes for the classification task. The default value is 1000,
  which is typically used for datasets like ImageNet.
