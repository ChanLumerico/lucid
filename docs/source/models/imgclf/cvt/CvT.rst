CvT
===

.. toctree::
    :maxdepth: 1
    :hidden:

    cvt_13.rst
    cvt_21.rst
    cvt_w24.rst

|transformer-badge| |vision-transformer-badge| |imgclf-badge|

.. autoclass:: lucid.models.CvT

The `CvT` (Convolutional Vision Transformer) class implements a hybrid vision transformer
that integrates convolutional layers into the self-attention mechanism. Unlike traditional
Vision Transformers (ViTs), CvT introduces depthwise convolutional projections in the query,
key, and value transformations, which enhances inductive biases and improves efficiency.
This hybrid approach helps in capturing both local and global features effectively while
reducing computational cost.

.. mermaid::
    :name: CvT

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>cvt_13</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_VisionTransformer x 3"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m1_in(["Input"]);
          m1_out(["Output"]);
      style m1_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m1_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
          subgraph sg_m2["_ConvEmbed"]
          style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,64,56,56)</span>"];
            m4["LayerNorm"];
          end
          m5["Dropout"];
          subgraph sg_m6["blocks"]
          style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m7["_ConvTransformerBlock"];
          end
        end
        m8["LayerNorm"];
        m9["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,384) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m3 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m4 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m5 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m8 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m9 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input -.-> m3;
      m1_in -.-> m3;
      m1_out -.-> m1_in;
      m1_out --> m8;
      m3 --> m4;
      m4 --> m5;
      m5 --> m7;
      m7 -.-> m1_in;
      m7 --> m1_out;
      m8 --> m9;
      m9 --> output;

Class Signature
---------------

.. code-block:: python

    class CvT(
        in_channels: int = 3,
        num_classes: int = 1000,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        spec: CvTSpec | None = None,
    )

Parameters
----------
- **in_channels** (*int*):
  Number of input channels (e.g., 3 for RGB images).

- **num_classes** (*int*):
  Number of output classes for classification.

- **act_layer** (*Type[nn.Module]*):
  Activation function used in the model (default is `nn.GELU`).

- **norm_layer** (*Type[nn.Module]*):
  Normalization layer to be used (default is `nn.LayerNorm`).

- **spec** (*CvTSpec | None*):
  Optional model specification defining the architecture details.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> cvt = models.CvT(
    ...     in_channels=3,
    ...     num_classes=1000,
    ... )

