ConvNeXt_V2
===========

.. toctree::
    :maxdepth: 1
    :hidden:

    convnext_v2_atto.rst
    convnext_v2_femto.rst
    convnext_v2_pico.rst
    convnext_v2_nano.rst
    convnext_v2_tiny.rst
    convnext_v2_base.rst
    convnext_v2_large.rst
    convnext_v2_huge.rst

|convnet-badge| 

.. autoclass:: lucid.models.ConvNeXt_V2

The `ConvNeXt_V2` module in `lucid.nn` builds upon the original ConvNeXt architecture, 
offering enhanced flexibility and efficiency. It introduces updated configurations for 
modern image classification tasks, while maintaining the hierarchical design of 
its predecessor.

.. mermaid::
    :name: ConvNeXt-v2

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>convnext_v2_base</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["downsample_layers"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m2["Sequential"]
          style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,128,56,56)</span>"];
            m4["_ChannelsFisrtLayerNorm"];
          end
          subgraph sg_m5["Sequential x 3"]
          style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m5_in(["Input"]);
            m5_out(["Output"]);
      style m5_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m5_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m6["_ChannelsFisrtLayerNorm"];
            m7["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,128,56,56) → (1,256,28,28)</span>"];
          end
        end
        subgraph sg_m8["stages"]
        style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m9["Sequential x 4"]
          style sg_m9 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m9_in(["Input"]);
            m9_out(["Output"]);
      style m9_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m9_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m10(["_Block_V2 x 3"]);
          end
        end
        m11["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,1024,7,7) → (1,1024,1,1)</span>"];
        m12["LayerNorm"];
        m13["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,1024) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m3 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m7 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m11 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m12 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m13 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m3;
      m10 -.-> m6;
      m10 --> m9_out;
      m11 --> m12;
      m12 --> m13;
      m13 --> output;
      m3 --> m4;
      m4 -.-> m10;
      m5_in -.-> m6;
      m5_out -.-> m9_in;
      m6 --> m7;
      m7 --> m5_out;
      m7 -.-> m9_in;
      m9_in -.-> m10;
      m9_out --> m11;
      m9_out --> m5_in;

Class Signature
---------------

.. code-block:: python

    class ConvNeXt_V2(ConvNeXt):
        def __init__(
            num_classes: int = 1000,
            depths: list[int] = [3, 3, 9, 3],
            dims: list[int] = [96, 192, 384, 768],
            drop_path: float = 0.0,
        )

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **depths** (*list[int]*, optional):
  The number of layers in each stage of the network. Default is [3, 3, 9, 3].

- **dims** (*list[int]*, optional):
  The number of channels in each stage of the network. Default is [96, 192, 384, 768].

- **drop_path** (*float*, optional):
  The stochastic depth drop path rate. Default is 0.0.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.models as models

    # Create ConvNeXt_V2 with default 1000 classes
    model = models.ConvNeXt_V2(num_classes=1000)

    # Input tensor with shape (1, 3, 224, 224)
    input_ = lucid.random.randn(1, 3, 224, 224)

    # Perform forward pass
    output = model(input_)
    print(output.shape)  # Shape: (1, 1000)
