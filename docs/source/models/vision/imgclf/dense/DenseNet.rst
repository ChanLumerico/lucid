DenseNet
========

.. toctree::
    :maxdepth: 1
    :hidden:

    DenseNetConfig.rst
    densenet_121.rst
    densenet_169.rst
    densenet_201.rst
    densenet_264.rst

|convnet-badge|

.. autoclass:: lucid.models.DenseNet

The `DenseNet` class implements the Dense Convolutional Network family with dense
feature reuse across each block. Model structure is defined through `DenseNetConfig`,
which captures the dense block depths, growth rate, initial stem width, and classifier
settings together with optional input-channel and transition-compression overrides.

.. mermaid::
    :name: DenseNet

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>densenet_121</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["ConvBNReLU2d"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,64,112,112)</span>"];
        m3["BatchNorm2d"];
        m4["ReLU"];
        end
        m5["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"];
        subgraph sg_m6["blocks"]
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m7(["_DenseBlock x 4<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"]);
        end
        subgraph sg_m8["transitions"]
        style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m9(["_TransitionLayer x 3<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56) → (1,128,28,28)</span>"]);
        end
        m10["BatchNorm2d"];
        m11["ReLU"];
        m12["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,1024,7,7) → (1,1024,1,1)</span>"];
        m13["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,1024) → (1,1000)</span>"];
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
    style m11 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m12 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
    style m13 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input --> m2;
    m10 --> m11;
    m11 --> m12;
    m12 --> m13;
    m13 --> output;
    m2 --> m3;
    m3 --> m4;
    m4 --> m5;
    m5 -.-> m7;
    m7 --> m10;
    m7 --> m9;
    m9 -.-> m7;

Class Signature
---------------

.. code-block:: python

    class DenseNet(nn.Module):
        def __init__(self, config: DenseNetConfig)

Parameters
----------

- **config** (*DenseNetConfig*):
  Configuration object describing the dense block depths, growth rate, stem width,
  classifier size, and optional input-channel, bottleneck, and compression settings.

Attributes
----------

- **config** (*DenseNetConfig*):
  The configuration used to construct the model.
- **growth_rate** (*int*):
  Number of new channels contributed by each dense layer.
- **num_init_features** (*int*):
  Output width of the initial convolution stem.
- **bottleneck** (*int*):
  Expansion factor used by the 1x1 bottleneck convolution inside each dense layer.
- **compression** (*float*):
  Transition-layer compression ratio applied between dense blocks.
- **conv0**, **pool0**, **blocks**, **transitions**, **bn_final**, **relu**, **avgpool**, **fc**:
  DenseNet stem, dense blocks, transition layers, normalization head, and classifier.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.models as models
    >>> config = models.DenseNetConfig(
    ...     block_config=[6, 12, 24, 16],
    ...     growth_rate=32,
    ...     num_init_features=64,
    ...     num_classes=10,
    ...     in_channels=1,
    ... )
    >>> model = models.DenseNet(config)
    >>> output = model(lucid.zeros(1, 1, 224, 224))
    >>> print(output.shape)
    (1, 10)

.. note::

   - Dense connectivity improves feature reuse and gradient flow across very deep stacks.
   - Factory helpers such as `densenet_121` and `densenet_201` provide standard ImageNet presets.
