InceptionNeXt
=============

.. toctree::
    :maxdepth: 1
    :hidden:

    InceptionNeXtConfig.rst
    inception_next_atto.rst
    inception_next_tiny.rst
    inception_next_small.rst
    inception_next_base.rst

|convnet-badge| 

.. autoclass:: lucid.models.InceptionNeXt

The `InceptionNeXt` class implements a modernized version of the Inception architecture,
leveraging depthwise convolutions, multi-layer perceptrons (MLPs), and advanced
token mixing techniques for enhanced performance and efficiency. Model structure
is defined through `InceptionNeXtConfig`.

.. mermaid::
    :name: InceptionNeXt

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>inception_next_base</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["stem"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,128,56,56)</span>"];
          m3["BatchNorm2d"];
        end
        subgraph sg_m4["stages"]
          direction TB;
        style sg_m4 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m5["_IncepNeXtStage"]
            direction TB;
          style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m6["Identity"];
            m7["Sequential"];
          end
          subgraph sg_m8["_IncepNeXtStage x 3"]
            direction TB;
          style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8_in(["Input"]);
            m8_out(["Output"]);
      style m8_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m8_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m9(["Sequential x 2<br/><span style='font-size:11px;font-weight:400'>(1,128,56,56) → (1,256,28,28)</span>"]);
          end
        end
        subgraph sg_m10["_MLPHead"]
          direction TB;
        style sg_m10 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m11["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,1024) → (1,3072)</span>"];
          m12["GELU"];
          m13["LayerNorm"];
          m14["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,1024,7,7) → (1,1024,1,1)</span>"];
          m15["Dropout"];
          m16["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,3072) → (1,1000)</span>"];
        end
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m6 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m11 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m12 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m13 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m14 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m15 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m16 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m11 --> m12;
      m12 --> m13;
      m13 --> m15;
      m14 --> m11;
      m15 --> m16;
      m16 --> output;
      m2 --> m3;
      m3 --> m6;
      m6 --> m7;
      m7 -.-> m9;
      m8_in -.-> m9;
      m8_out --> m14;
      m8_out -.-> m8_in;
      m9 -.-> m8_in;
      m9 --> m8_out;

Class Signature
---------------

.. code-block:: python

    class InceptionNeXt(nn.Module):
        def __init__(self, config: InceptionNeXtConfig) -> None

Parameters
----------

- **config** (*InceptionNeXtConfig*):
  Configuration object describing stage depths, stage widths, token mixers,
  MLP ratios, classifier head settings, and regularization values.

Examples
--------

.. code-block:: python

    from lucid.models import InceptionNeXt, InceptionNeXtConfig

    config_tiny = InceptionNeXtConfig(
        num_classes=1000,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
    )
    model_tiny = InceptionNeXt(config_tiny)

    input_tensor = lucid.random.randn(1, 3, 224, 224)
    output = model_tiny(input_tensor)
    print(output.shape)
