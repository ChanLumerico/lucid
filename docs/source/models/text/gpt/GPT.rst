GPT
===

.. toctree::
    :maxdepth: 1
    :hidden:

    GPTConfig.rst
    GPTTokenizerFast.rst
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
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>GPT</span>"]
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
    end
    input["Input"];
    output["Output"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m2 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m3 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m7 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m9 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    input --> m2;
    m10 -.-> m6_in;
    m10 --> m6_out;
    m2 --> m3;
    m3 -.-> m7;
    m6_in -.-> m7;
    m6_out -.-> m6_in;
    m6_out --> output;
    m7 --> m8;
    m8 --> m9;
    m9 --> m10;

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
