GPT2
====

.. toctree::
    :maxdepth: 1
    :hidden:

    GPT2Config.rst
    GPT2TokenizerFast.rst
    GPT2LMHeadModel.rst
    GPT2DoubleHeadsModel.rst
    GPT2ForSequenceClassification.rst

    gpt2_small.rst
    gpt2_medium.rst
    gpt2_large.rst
    gpt2_xlarge.rst

|transformer-badge| |decoder-only-transformer-badge|

.. autoclass:: lucid.models.GPT2

The `GPT2` class provides a decoder-only causal Transformer backbone following
the GPT-2 architecture. It extends GPT-1 with a final LayerNorm (`ln_f`)
applied after the decoder stack, a larger vocabulary (50257 tokens via Byte-level BPE),
a longer context window (1024 tokens).

.. mermaid::
    :name: GPT2

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>GPT2</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["_GPTEmbedding"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m2(["Embedding x 2"]);
        m3["Dropout"];
        end
        subgraph sg_m4["_GPTDecoder"]
        style sg_m4 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m5["h"]
            direction TB;
        style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m6["_GPTBlock x 12"]
            direction TB;
            style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m6_in(["Input"]);
            m6_out(["Output"]);
    style m6_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
    style m6_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
            m7["LayerNorm"];
            m8["_GPTAttention"];
            m9["LayerNorm"];
            m10["_GPTMLP"];
            end
        end
        end
        m11["LayerNorm"];
    end
    input["Input"];
    output["Output"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m2 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m3 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m7 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m9 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m11 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    input --> m2;
    m10 -.-> m6_in;
    m10 --> m6_out;
    m11 --> output;
    m2 --> m3;
    m3 -.-> m7;
    m6_in -.-> m7;
    m6_out --> m11;
    m6_out -.-> m6_in;
    m7 --> m8;
    m8 --> m9;
    m9 --> m10;

Class Signature
---------------

.. code-block:: python

    class GPT2(config: GPT2Config)

Parameters
----------
- **config** (*GPT2Config*):
  Configuration object defining vocabulary size, hidden dimensions, depth,
  attention setup, and runtime behavior (causal masking, cache).

Methods
-------

.. automethod:: lucid.models.GPT2.forward
.. automethod:: lucid.models.GPT2.get_input_embeddings
.. automethod:: lucid.models.GPT2.set_input_embeddings

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.GPT2Config.small()
    >>> model = models.GPT2(config)
    >>> print(model)
    GPT2(...)

.. code-block:: python

    >>> import lucid
    >>> input_ids = lucid.randint(0, config.vocab_size, (1, 16))
    >>> hidden_states, _ = model(input_ids)
    >>> hidden_states.shape
    (1, 16, 768)
