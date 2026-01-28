Xception
========

.. toctree::
    :maxdepth: 1
    :hidden:

    xception.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.Xception

The `Xception` class implements the base Xception architecture, 
leveraging depthwise separable convolutions for efficient and scalable feature extraction. 
This model is designed for classification tasks and provides flexibility 
for the number of output classes.

.. mermaid::
    :name: Xception

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>xception</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m1["ReLU"];
        subgraph sg_m2["ConvBNReLU2d x 2"]
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m2_in(["Input"]);
        m2_out(["Output"]);
    style m2_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m2_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
        m3["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,32,111,111)</span>"];
        m4["BatchNorm2d"];
        m5["ReLU"];
        end
        subgraph sg_m6["_Block x 3"]
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m6_in(["Input"]);
        m6_out(["Output"]);
    style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
        m7["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,64,109,109) → (1,128,55,55)</span>"];
        m8["BatchNorm2d"];
        m9["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,109,109) → (1,128,55,55)</span>"];
        end
        subgraph sg_m10["mid_blocks"]
        style sg_m10 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m11(["_Block x 8"]);
        end
        subgraph sg_m12["_Block"]
        style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m13["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,728,14,14) → (1,1024,7,7)</span>"];
        m14["BatchNorm2d"];
        m15["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,728,14,14) → (1,1024,7,7)</span>"];
        end
        subgraph sg_m16["DepthSeparableConv2d"]
        style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m17(["Conv2d x 2"]);
        end
        m18["BatchNorm2d"];
        subgraph sg_m19["DepthSeparableConv2d"]
        style sg_m19 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m20(["Conv2d x 2"]);
        end
        m21["BatchNorm2d"];
        m22["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,2048,7,7) → (1,2048,1,1)</span>"];
        m23["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2048) → (1,1000)</span>"];
    end
    input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
    output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m1 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m3 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m4 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m5 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m7 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m8 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m13 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m14 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m17 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m18 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m20 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m21 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m22 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
    style m23 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input -.-> m3;
    m1 --> m20;
    m1 --> m22;
    m11 --> m15;
    m13 --> m14;
    m14 --> m17;
    m15 --> m13;
    m17 --> m18;
    m18 -.-> m1;
    m20 --> m21;
    m21 -.-> m1;
    m22 --> m23;
    m23 --> output;
    m2_in -.-> m3;
    m2_out --> m9;
    m3 --> m4;
    m4 --> m5;
    m5 --> m2_in;
    m5 --> m2_out;
    m6_in -.-> m7;
    m6_out --> m11;
    m6_out -.-> m6_in;
    m7 --> m8;
    m8 -.-> m6_in;
    m9 --> m6_out;
    m9 -.-> m7;

Class Signature
---------------

.. code-block:: python

    class lucid.models.Xception(num_classes: int = 1000) -> None

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

Attributes
----------

- **features** (*Module*):
  The sequential feature extractor composed of depthwise separable convolutions.

- **classifier** (*Module*):
  The fully connected classification head.

Forward Pass
------------

The forward method applies the feature extractor to the input tensor, 
followed by the classifier for prediction. 

The overall structure can be summarized as:

.. math::

    \mathbf{y} = \text{classifier}(\text{features}(\mathbf{x}))

Where:
- :math:`\mathbf{x}` is the input tensor.
- :math:`\text{features}` is the feature extraction module.
- :math:`\text{classifier}` is the fully connected head.
- :math:`\mathbf{y}` is the output tensor.

Examples
--------

**Basic Usage:**

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.Xception(num_classes=1000)
    >>> input_tensor = lucid.random.randn(1, 3, 299, 299)  # Example input with shape (batch_size, channels, height, width)
    >>> output = model(input_tensor)
    >>> print(output.shape)
    (1, 1000)

**Modifying Output Classes:**

.. code-block:: python

    >>> model = models.Xception(num_classes=1000)
    >>> print(model)
    ...
