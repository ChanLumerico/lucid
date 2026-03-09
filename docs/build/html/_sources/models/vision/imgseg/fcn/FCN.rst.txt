FCN
===
|convnet-badge| |segmentation-convnet-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    FCNConfig.rst
    fcn_resnet_50.rst
    fcn_resnet_101.rst

.. autoclass:: lucid.models.FCN

`FCN` is a fully convolutional semantic segmentation model built around a
ResNet feature extractor and lightweight classifier heads. The backbone
produces dense feature maps, the main classifier predicts per-class logits,
and an optional auxiliary classifier provides intermediate supervision from a
shallower stage.

.. mermaid::
    :name: FCN

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>fcn_resnet_50</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_FCNResNetBackbone"]
        direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m2["Conv2d"];
        m3["BatchNorm2d"];
        m4["ReLU"];
        m5["MaxPool2d"];
        subgraph sg_m6["layer1 x 4"]
            direction TB;
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m6_in(["Input"]);
            m6_out(["Output"]);
    style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m7(["_Bottleneck x 3"]);
        end
        end
        subgraph sg_m8["classifier x 2"]
        direction TB;
        style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m8_in(["Input"]);
        m8_out(["Output"]);
    style m8_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m8_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
        m9["Conv2d"];
        m10["BatchNorm2d"];
        m11["ReLU"];
        m12["Dropout"];
        m13["Conv2d"];
        end
    end
    input["Input"];
    output["Output"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m5 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
    style m9 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    style m10 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m11 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m12 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m13 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
    input --> m2;
    m10 --> m11;
    m11 --> m12;
    m12 --> m13;
    m13 --> m8_in;
    m13 --> m8_out;
    m2 --> m3;
    m3 --> m4;
    m4 --> m5;
    m5 -.-> m7;
    m6_in -.-> m7;
    m6_out -.-> m6_in;
    m6_out -.-> m9;
    m7 -.-> m6_in;
    m7 --> m6_out;
    m8_in -.-> m9;
    m8_out --> output;
    m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class FCN(_FCNSegmentationModel, PreTrainedModelMixin):
        def __init__(self, config: FCNConfig) -> None

Parameters
----------

- **config** (*FCNConfig*):
  Model configuration describing the backbone choice, classifier widths, and
  segmentation output space.

Methods
-------

.. automethod:: lucid.models.FCN.forward

Examples
--------

**Build from a Preset Builder**

.. code-block:: python

    from lucid.models import fcn_resnet_50
    import lucid

    model = fcn_resnet_50(num_classes=21)
    x = lucid.random.randn(1, 3, 224, 224)

    logits = model(x)
    print(logits.shape)

**Return Auxiliary Output**

.. code-block:: python

    from lucid.models import fcn_resnet_50
    import lucid

    model = fcn_resnet_50(num_classes=21, aux_loss=True)
    x = lucid.random.randn(1, 3, 224, 224)

    out = model(x, return_aux=True)
    print(out["out"].shape)
    print(out["aux"].shape)
