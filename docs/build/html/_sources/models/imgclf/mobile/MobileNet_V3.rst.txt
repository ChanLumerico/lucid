MobileNet_V3
============

.. toctree::
    :maxdepth: 1
    :hidden:

    mobilenet_v3_small.rst
    mobilenet_v3_large.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.MobileNet_V3

Overview
--------

The `MobileNetV3` class implements the MobileNet-v3 architecture, 
building upon the innovations of MobileNet-v2 with further optimizations. 
It introduces **squeeze-and-excitation modules**, **efficient head designs**, 
and two variants—**Small** and **Large**—tailored for different resource constraints. 

This architecture is designed for high performance in mobile and embedded applications 
with minimal computational overhead.

.. mermaid::
    :name: MobileNetV3

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>mobilenet_v3_small</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["conv_first"]
        direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,16,112,112)</span>"];
        m3["BatchNorm2d"];
        m4["HardSwish"];
        end
        subgraph sg_m5["bottlenecks"]
        direction TB;
        style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m6["_InvertedBottleneck_V3 x 11"]
            direction TB;
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m6_in(["Input"]);
            m6_out(["Output"]);
    style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m7["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,16,112,112) → (1,16,56,56)</span>"];
        end
        end
        subgraph sg_m8["conv_last"]
        direction TB;
        style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m9["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,96,7,7) → (1,576,7,7)</span>"];
        m10["BatchNorm2d"];
        m11["HardSwish"];
        end
        m12["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,576,7,7) → (1,576,1,1)</span>"];
        subgraph sg_m13["fc1"]
        direction TB;
        style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m14["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,576) → (1,1024)</span>"];
        m15["HardSwish"];
        end
        subgraph sg_m16["fc2"]
        direction TB;
        style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m17["Dropout"];
        m18["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,1024) → (1,1000)</span>"];
        end
    end
    input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
    output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m9 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m10 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m11 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m12 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
    style m14 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m15 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m17 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m18 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input --> m2;
    m10 --> m11;
    m11 --> m12;
    m12 --> m14;
    m14 --> m15;
    m15 --> m17;
    m17 --> m18;
    m18 --> output;
    m2 --> m3;
    m3 --> m4;
    m4 -.-> m7;
    m6_in -.-> m7;
    m6_out -.-> m6_in;
    m6_out --> m9;
    m7 -.-> m6_in;
    m7 --> m6_out;
    m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class MobileNet_V3(nn.Module):
        def __init__(
            self, bottleneck_cfg: list, last_channels: int, num_classes: int = 1000
        ) -> None

Parameters
----------
- **bottleneck_cfg** (*list*):  
  Configuration of bottleneck layers, defining the structure and number 
  of each inverted residual block.

- **last_channels** (*int*):  
  Number of channels in the final convolutional layer.

- **num_classes** (*int*, optional):  
  Number of output classes for the classification task. Default is 1000, 
  commonly used for ImageNet.

Examples
--------

**Creating a MobileNetV3 model with custom configurations:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> bottleneck_cfg = [
    ...     # (kernel_size, expansion_factor, out_channels, stride)
    ...     (3, 16, 16, 1),
    ...     (3, 64, 24, 2),
    ...     (5, 72, 40, 2),
    ...     # Additional layers can be added as per requirement
    ... ]
    >>> model = nn.MobileNet_V3(bottleneck_cfg=bottleneck_cfg, last_channels=1280, num_classes=1000)
    >>> print(model)

**Forward pass with MobileNetV3:**

.. code-block:: python

    >>> from lucid.tensor import Tensor
    >>> input_tensor = Tensor([[...]])  # Input tensor with appropriate shape
    >>> output = model(input_tensor)
    >>> print(output)

.. note::

   The MobileNetV3 architecture is optimized for resource efficiency and high 
   performance on edge devices. Depending on your use case, you can adjust the 
   bottleneck configurations and final output channels to balance accuracy and 
   computational cost.
