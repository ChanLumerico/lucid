ResNeXt
=======

.. toctree::
    :maxdepth: 1
    :hidden:

    ResNeXtConfig.rst
    resnext_50_32x4d.rst
    resnext_101_32x4d.rst
    resnext_101_32x8d.rst
    resnext_101_32x16d.rst
    resnext_101_32x32d.rst
    resnext_101_64x4d.rst

|convnet-badge| 

.. autoclass:: lucid.models.ResNeXt

The `ResNeXt` class extends the `ResNet` family with grouped bottleneck convolutions,
using cardinality and base width as the main scaling knobs. Model structure is defined
through `ResNeXtConfig`, which captures the stage depths together with grouped-convolution
hyperparameters and the shared ResNet stem options.

.. mermaid::
    :name: ResNeXt

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>resnext_50_32x4d</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["stem"]
          direction TB;
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m2["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,64,112,112)</span>"];
          m3["BatchNorm2d"];
          m4["ReLU"];
        end
        m5["MaxPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,64,112,112) → (1,64,56,56)</span>"];
        subgraph sg_m6["layer1 x 4"]
          direction TB;
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m6_in(["Input"]);
          m6_out(["Output"]);
      style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m7["_Bottleneck"]
            direction TB;
          style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8(["ConvBNReLU2d x 2<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,128,56,56)</span>"]);
            m9["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,128,56,56) → (1,256,56,56)</span>"];
            m10["ReLU"];
            m11["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,64,56,56) → (1,256,56,56)</span>"];
          end
          subgraph sg_m12["_Bottleneck x 2"]
            direction TB;
          style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m12_in(["Input"]);
            m12_out(["Output"]);
      style m12_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m12_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m13(["ConvBNReLU2d x 2<br/><span style='font-size:11px;font-weight:400'>(1,256,56,56) → (1,128,56,56)</span>"]);
            m14["Sequential<br/><span style='font-size:11px;font-weight:400'>(1,128,56,56) → (1,256,56,56)</span>"];
            m15["ReLU"];
          end
        end
        m16["AdaptiveAvgPool2d<br/><span style='font-size:11px;color:#b7791f;font-weight:400'>(1,2048,7,7) → (1,2048,1,1)</span>"];
        m17["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,2048) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m2 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m4 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m5 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m10 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m15 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
      style m16 fill:#fefcbf,stroke:#b7791f,stroke-width:1px;
      style m17 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m2;
      m10 -.-> m13;
      m11 --> m10;
      m12_in -.-> m13;
      m12_out -.-> m6_in;
      m13 --> m14;
      m14 --> m15;
      m15 --> m12_in;
      m15 --> m12_out;
      m15 --> m6_out;
      m16 --> m17;
      m17 --> output;
      m2 --> m3;
      m3 --> m4;
      m4 --> m5;
      m5 -.-> m8;
      m6_in -.-> m8;
      m6_out --> m16;
      m6_out -.-> m6_in;
      m8 --> m9;
      m9 --> m11;

Class Signature
---------------

.. code-block:: python

    class ResNeXt(ResNet):
        def __init__(self, config: ResNeXtConfig)

Parameters
----------

- **config** (*ResNeXtConfig*):
  Configuration object describing the stage depths, grouped convolution hyperparameters,
  classifier size, stem settings, and any extra bottleneck keyword arguments.

Attributes
----------

- **config** (*ResNeXtConfig*):
  The configuration used to construct the model.
- **cardinality** (*int*):
  Number of groups used in grouped bottleneck convolutions.
- **base_width** (*int*):
  Width per group used by the bottleneck blocks.
- **stem**, **layer1**, **layer2**, **layer3**, **layer4**, **avgpool**, **fc**:
  Inherited ResNet submodules assembled from the configuration.

Forward Calculation
-------------------

The forward pass of the `ResNeXt` model includes:

1. **Stem**: Initial convolutional layers for feature extraction.
2. **Grouped Bottleneck Stages**: Residual stages built with grouped bottleneck blocks.
3. **Global Pooling**: A global average pooling layer reduces spatial dimensions.
4. **Classifier**: A fully connected layer maps the features to class scores.

.. math::

    \text{output} = \text{FC}(\text{GAP}(\text{GroupedConvBlocks}(\text{Stem}(\text{input}))))

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.models as models
    >>> config = models.ResNeXtConfig(
    ...     layers=[3, 4, 6, 3],
    ...     cardinality=32,
    ...     base_width=4,
    ...     num_classes=10,
    ...     in_channels=1,
    ...     stem_type="deep",
    ...     stem_width=32,
    ... )
    >>> model = models.ResNeXt(config)
    >>> output = model(lucid.zeros(1, 1, 224, 224))
    >>> print(output.shape)
    (1, 10)

.. note::

   - `ResNeXt` introduces cardinality as a scaling axis for grouped bottleneck blocks.
   - Factory helpers such as `resnext_50_32x4d` and `resnext_101_64x4d` provide standard presets.
