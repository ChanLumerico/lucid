Transformer
===========

.. toctree::
    :maxdepth: 1
    :hidden:

    transformer_base.rst
    transformer_big.rst

|transformer-badge|

.. autoclass:: lucid.models.Transformer

The `Transformer` class in `model` provides a full implementation of the Transformer model,
including positional encoding and the final vocabulary projection. This is distinct from `nn.Transformer`,
which serves as a generic module template for building Transformer components.

.. mermaid::
    :name: Transformer

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>transformer_base</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m1(["Embedding x 2<br/><span style='font-size:11px;color:#475569;font-weight:400'>(1,100) → (1,100,512)</span>"]);
        subgraph sg_m2["_PositionalEncoding"]
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          m3["Dropout"];
        end
        subgraph sg_m4["Transformer"]
        style sg_m4 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m5["TransformerEncoder"]
          style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m6["layers"]
            style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              subgraph sg_m7["TransformerEncoderLayer x 6"]
                direction TB;
              style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
                m7_in(["Input"]);
                m7_out(["Output"]);
      style m7_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m7_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
                subgraph sg_m8["MultiHeadAttention"]
                  direction TB;
                style sg_m8 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
                  m9(["Linear x 4"]);
                end
                m10(["Linear x 2<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,100,512) → (1,100,2048)</span>"]);
                m11(["Dropout x 3"]);
                m12(["LayerNorm x 2"]);
              end
            end
            m13["LayerNorm"];
          end
          subgraph sg_m14["TransformerDecoder"]
          style sg_m14 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m15["layers"]
            style sg_m15 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              subgraph sg_m16["TransformerDecoderLayer x 6"]
                direction TB;
              style sg_m16 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
                m16_in(["Input"]);
                m16_out(["Output"]);
      style m16_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m16_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
                subgraph sg_m17["MultiHeadAttention x 2"]
                  direction TB;
                style sg_m17 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
                  m17_in(["Input"]);
                  m17_out(["Output"]);
      style m17_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m17_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
                  m18(["Linear x 4"]);
                end
                m19(["Linear x 2<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,100,512) → (1,100,2048)</span>"]);
                m20(["Dropout x 4"]);
                m21(["LayerNorm x 3"]);
              end
            end
            m22["LayerNorm"];
          end
        end
        m23["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,100,512) → (1,100,12000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,100)x2</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,100,12000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m1 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
      style m3 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m9 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m10 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m11 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m12 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m13 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m18 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m19 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m20 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m21 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m22 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m23 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m1;
      m1 --> m3;
      m10 -.-> m11;
      m11 -.-> m10;
      m11 --> m12;
      m12 -.-> m10;
      m12 -.-> m7_in;
      m12 --> m7_out;
      m13 -.-> m18;
      m16_in -.-> m18;
      m16_out -.-> m16_in;
      m16_out --> m22;
      m17_in -.-> m18;
      m17_out -.-> m20;
      m18 --> m17_out;
      m18 -.-> m20;
      m19 -.-> m20;
      m20 -.-> m19;
      m20 --> m21;
      m21 -.-> m16_in;
      m21 --> m16_out;
      m21 --> m17_in;
      m21 -.-> m19;
      m22 --> m23;
      m23 --> output;
      m3 -.-> m9;
      m7_in -.-> m9;
      m7_out --> m13;
      m7_out -.-> m7_in;
      m9 -.-> m11;

Class Signature
---------------

.. code-block:: python

    class Transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    )

Parameters
----------
- **src_vocab_size** (*int*):
  Size of the source vocabulary.

- **tgt_vocab_size** (*int*):
  Size of the target vocabulary.

- **d_model** (*int*):
  Dimension of the model’s hidden representations.

- **num_heads** (*int*):
  Number of attention heads in the multi-head self-attention mechanism.

- **num_encoder_layers** (*int*):
  Number of encoder layers in the Transformer.

- **num_decoder_layers** (*int*):
  Number of decoder layers in the Transformer.

- **dim_feedforward** (*int*):
  Dimension of the feedforward network within each layer.

- **dropout** (*float*, optional):
  Dropout probability applied throughout the model. Default is 0.1.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> transformer = models.Transformer(
    ...     src_vocab_size=5000,
    ...     tgt_vocab_size=5000,
    ...     d_model=512,
    ...     num_heads=8,
    ...     num_encoder_layers=6,
    ...     num_decoder_layers=6,
    ...     dim_feedforward=2048,
    ...     dropout=0.1
    ... )
    >>> print(transformer)
    Transformer(src_vocab_size=5000, tgt_vocab_size=5000, d_model=512, ...)

This implementation follows the standard Transformer architecture and is ready to 
be trained for sequence-to-sequence tasks like machine translation.

Differences from `nn.Transformer`
----------------------------------
- This class implements a **complete Transformer model**, including **positional encoding** 
  and the **final projection** to vocabulary space.

- `nn.Transformer`, in contrast, provides a **modular base class** for 
  constructing Transformer components but does not include full integration.
