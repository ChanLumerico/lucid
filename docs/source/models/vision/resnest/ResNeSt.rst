ResNeSt
=======

.. toctree::
    :maxdepth: 1
    :hidden:

    resnest_14.rst
    resnest_26.rst
    resnest_50.rst
    resnest_101.rst
    resnest_200.rst
    resnest_269.rst
    resnest_50_4s2x40d.rst
    resnest_50_1s4x24d.rst

|convnet-badge| 

.. autoclass:: lucid.models.ResNeSt

The `ResNeSt` class extends the ResNet architecture by integrating Split-Attention blocks,
providing improved representational power for a variety of vision tasks.

.. mermaid::
    :name: ResNeSt

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>resnest_50</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["stem"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,32,112,112)</span>"];
          m3["BatchNorm2d"];
          m4["ReLU"];
          m5["Conv2d"];
          m6["BatchNorm2d"];
          m7["ReLU"];
          m8["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,32,112,112) → (1,64,112,112)</span>"];
        end
        m9["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"];
        subgraph sg_m10["layer1 x 4"]
        style sg_m10 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m10_in(["Input"]);
          m10_out(["Output"]);
      style m10_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m10_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          m11(["_ResNeStBottleneck x 3<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"]);
        end
        m12["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,2048,7,7) → (1,2048,1,1)</span>"];
        m13["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2048) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m5 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m6 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m7 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m8 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m9 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m12 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m13 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10_in -.-> m11;
      m10_out -.-> m10_in;
      m10_out --> m12;
      m11 -.-> m10_in;
      m11 --> m10_out;
      m12 --> m13;
      m13 --> output;
      m2 --> m3;
      m3 --> m4;
      m4 --> m5;
      m5 --> m6;
      m6 --> m7;
      m7 --> m8;
      m8 --> m9;
      m9 -.-> m11;

Class Signature
---------------

.. code-block:: python

    class ResNeSt(ResNet):
        def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            base_width: int = 64,
            stem_width: int = 32,
            cardinality: int = 1,
            radix: int = 2,
            avd: bool = True,
        ) -> None

Parameters
----------
- **block** (*type*):
  The block class to be used in constructing the network (e.g., `_ResNeStBottleneck`).

- **layers** (*list[int]*):
  A list specifying the number of blocks at each stage of the network.

- **num_classes** (*int*, optional):
  Number of output classes for the classification task. Defaults to 1000.

- **base_width** (*int*, optional):
  Base width of the network for scaling the number of channels. Defaults to 64.

- **stem_width** (*int*, optional):
  Width of the initial stem convolution. Defaults to 32.

- **cardinality** (*int*, optional):
  Number of groups for grouped convolutions. Defaults to 1.

- **radix** (*int*, optional):
  Number of splits for the Split-Attention block. Defaults to 2.

- **avd** (*bool*, optional):
  Whether to use Adaptive Average Pooling for downsampling. Defaults to True.

Examples
--------

.. code-block:: python

    from lucid.models import ResNeSt

    # Define a ResNeSt model with 50 layers
    model = ResNeSt(
        block=_ResNeStBottleneck,
        layers=[3, 4, 6, 3],  # ResNet-50 configuration
        num_classes=1000,
        base_width=64,
        stem_width=32,
        cardinality=1,
        radix=2,
        avd=True
    )

    # Forward pass with a sample input
    input_tensor = lucid.random.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Output: (1, 1000)

.. note::

    The `ResNeSt` class enables flexibility in defining variants of ResNeSt models 
    (e.g., ResNeSt-50, ResNeSt-101, etc.) by customizing parameters such as `layers` 
    and `block`.