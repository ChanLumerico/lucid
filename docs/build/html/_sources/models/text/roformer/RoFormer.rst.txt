RoFormer
========

.. toctree::
    :maxdepth: 1
    :hidden:

    RoFormerConfig.rst
    RoFormerForMaskedLM.rst
    RoFormerForSequenceClassification.rst
    RoFormerForTokenClassification.rst
    RoFormerForMultipleChoice.rst
    RoFormerForQuestionAnswering.rst

|transformer-badge| |encoder-only-transformer-badge|

.. autoclass:: lucid.models.RoFormer

The `RoFormer` class provides a BERT-compatible encoder backbone that applies
rotary position embeddings (RoPE) in self-attention.

.. mermaid::
    :name: RoFormer

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>RoFormerForMaskedLM</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["RoFormer"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m2["_RoFormerEmbeddings"]
            direction TB;
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3(["Embedding x 2"]);
            m4["LayerNorm"];
            m5["Dropout"];
        end
        subgraph sg_m6["_RoFormerEncoder"]
            direction TB;
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m7["layer"]
            direction TB;
            style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8(["_RoFormerLayer x 12"]);
            end
        end
        subgraph sg_m9["_BERTPooler"]
            direction TB;
        style sg_m9 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m10["Linear"];
            m11["Tanh"];
        end
        m18["Linear"];
        end
        subgraph sg_m13["_BERTOnlyMLMHead"]
        style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m14["_BERTLMPredictionHead"]
            direction TB;
        style sg_m14 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m15["_BERTPredictionHeadTransform"]
            direction TB;
            style sg_m15 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m16["Linear"];
            m17["LayerNorm"];
            end
            m18["Linear"];
        end
        end
    end
    input["Input"];
    output["Output"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m3 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m4 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m5 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m10 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m11 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m18 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m16 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m17 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m18 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input --> m3;
    m10 --> m11;
    m11 --> m16;
    m16 --> m17;
    m17 --> m18;
    m18 --> output;
    m3 --> m4;
    m4 --> m5;
    m5 --> m8;
    m8 --> m10;

Class Signature
---------------

.. code-block:: python

    class RoFormer(config: RoFormerConfig)

Parameters
----------
- **config** (*RoFormerConfig*):
  Configuration object defining architecture, attention behavior, and RoPE
  options.

Methods
-------

.. automethod:: lucid.models.RoFormer.forward
.. automethod:: lucid.models.RoFormer.tie_weights
.. automethod:: lucid.models.RoFormer.get_input_embeddings
.. automethod:: lucid.models.RoFormer.set_input_embeddings
.. automethod:: lucid.models.RoFormer.get_output_embeddings
.. automethod:: lucid.models.RoFormer.set_output_embeddings

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.RoFormerConfig.base(
    ...     vocab_size=50000,
    ...     max_position_embeddings=512,
    ...     rope_interleaved=True,
    ... )
    >>> model = models.RoFormer(config)
    >>> print(model)
    RoFormer(...)
