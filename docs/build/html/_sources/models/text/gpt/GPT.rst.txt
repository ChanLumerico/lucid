GPT
===

.. toctree::
    :maxdepth: 1
    :hidden:

    GPTConfig.rst
    GPTLMHeadModel.rst
    GPTDoubleHeadsModel.rst
    GPTForSequenceClassification.rst

|transformer-badge| |decoder-only-transformer-badge|

.. autoclass:: lucid.models.GPT

The `GPT` class provides a decoder-only causal Transformer backbone
for autoregressive language modeling, following the original GPT-1 architecture.
It uses learned absolute position embeddings, Pre-LayerNorm blocks, and a
triangular causal mask for unidirectional attention.

.. mermaid::
    :name: GPT

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>gpt_lm_head_model</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["GPT"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m2["_GPTEmbedding"]
            direction TB;
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3(["Embedding x 2<br/><span style='font-size:11px;color:#475569;font-weight:400'>(1,8) → (1,8,768)</span>"]);
            m4["Dropout"];
        end
        subgraph sg_m5["_GPTDecoder"]
            direction TB;
        style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m6["h"]
            direction TB;
            style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
                subgraph sg_m7["_GPTBlock"]
                direction TB;
                style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
                    m8["LayerNorm"];
                    m9(["_GPTAttention<br/><span style='font-size:11px;font-weight:400'>(1,8,768) → (1,8,768)</span>"]);
                    m10["LayerNorm"];
                    m11(["_GPTMLP<br/><span style='font-size:11px;font-weight:400'>(1,8,768) → (1,8,768)</span>"]);
                end
            end
            m12(["_GPTBlock x 12<br/><span style='font-size:11px;font-weight:400'>(1,8,768) → (1,8,768)</span>"]);
        end
        end
        m13["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,8,768) → (1,8,40478)</span>"];
    end
    input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,8)</span>"];
    output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,8,40478)</span>"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m3 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m4 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m8 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m9 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m10 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m11 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m12 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m13 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input --> m3;
    m3 --> m4;
    m4 --> m12;
    m12 --> m13;
    m13 --> output;
    m8 --> m9;
    m9 --> m10;
    m10 --> m11;

Class Signature
---------------

.. code-block:: python

    class GPT(config: GPTConfig)

Parameters
----------
- **config** (*GPTConfig*):
  Configuration object defining vocabulary size, hidden dimensions, depth,
  attention setup, and runtime behavior (causal masking, cache).

Methods
-------

.. automethod:: lucid.models.GPT.forward
.. automethod:: lucid.models.GPT.get_input_embeddings
.. automethod:: lucid.models.GPT.set_input_embeddings

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.GPTConfig.base()
    >>> model = models.GPT(config)
    >>> print(model)
    GPT(...)

.. code-block:: python

    >>> import lucid
    >>> input_ids = lucid.randint(0, config.vocab_size, (1, 16))
    >>> hidden_states, _ = model(input_ids)
    >>> hidden_states.shape
    (1, 16, 768)
