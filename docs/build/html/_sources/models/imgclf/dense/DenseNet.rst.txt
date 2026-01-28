DenseNet
========

.. toctree::
    :maxdepth: 1
    :hidden:

    densenet_121.rst
    densenet_169.rst
    densenet_201.rst
    densenet_264.rst

|convnet-badge| |imgclf-badge|

.. autoclass:: lucid.models.DenseNet

The `DenseNet` class implements the Dense Convolutional Network (DenseNet),
featuring dense connections, bottleneck layers, and transition layers.

This class serves as the base for various DenseNet variants such as DenseNet-121, 
DenseNet-169, etc.

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

    class lucid.nn.DenseNet(
        block_config: tuple[int],
        growth_rate: int = 32,
        num_init_features: int = 64,
        num_classes: int = 1000,
    )

Parameters
----------
- **block_config** (*tuple[int]*): 
  Specifies the number of layers in each dense block.

- **growth_rate** (*int*, optional):
  Number of output channels added by each dense layer. Default is 32.

- **num_init_features** (*int*, optional):
  Number of output channels from the initial convolution layer. Default is 64.

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.

Examples
--------

**Defining a DenseNet-121 model:**

.. code-block:: python

    from lucid.models import DenseNet

    model = DenseNet(
        block_config=(6, 12, 24, 16),  # DenseNet-121
        growth_rate=32,
        num_init_features=64,
        num_classes=1000
    )

    input_tensor = lucid.random.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor)
    print(output.shape)  # Output shape: (1, 1000)

**Using a custom configuration:**

.. code-block:: python

    model = DenseNet(
        block_config=(4, 8, 16, 12),
        growth_rate=24,
        num_init_features=48,
        num_classes=10
    )

.. note::

  DenseNet is a memory-intensive architecture due to dense connections. 
  Consider optimizing for memory usage in large-scale applications.
