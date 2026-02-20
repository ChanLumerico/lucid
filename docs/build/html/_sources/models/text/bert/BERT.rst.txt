BERT
====

.. toctree::
    :maxdepth: 1
    :hidden:

    BERTConfig.rst
    BERTTokenizerFast.rst
    BERTForPreTraining.rst
    BERTForMaskedLM.rst
    BERTForCausalLM.rst
    BERTForNextSentencePrediction.rst
    BERTForSequenceClassification.rst
    BERTForTokenClassification.rst
    BERTForQuestionAnswering.rst

|transformer-badge| |encoder-only-transformer-badge|

.. autoclass:: lucid.models.BERT

The `BERT` class provides a configurable bidirectional Transformer backbone
for text representation learning and can be paired with multiple task heads.

.. mermaid::
    :name: BERT

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
    linkStyle default stroke-width:2.0px
    subgraph sg_m0["<span style='font-size:20px;font-weight:700'>bert_for_pre_training_base</span>"]
    style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m1["BERT"]
        style sg_m1 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m2["_BERTEmbeddings"]
            direction TB;
        style sg_m2 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m3(["Embedding x 3<br/><span style='font-size:11px;color:#475569;font-weight:400'>(1,8) → (1,8,768)</span>"]);
            m4["LayerNorm"];
            m5["Dropout"];
        end
        subgraph sg_m6["_BERTEncoder"]
            direction TB;
        style sg_m6 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m7["layer"]
            direction TB;
            style sg_m7 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m8(["_BERTLayer x 12<br/><span style='font-size:11px;font-weight:400'>(1,8,768)x2 → (1,8,768)</span>"]);
            end
        end
        subgraph sg_m9["_BERTPooler"]
            direction TB;
        style sg_m9 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m10["Linear"];
            m11["Tanh"];
        end
        end
        subgraph sg_m12["_BERTPreTrainingHeads"]
        style sg_m12 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        subgraph sg_m13["_BERTLMPredictionHead"]
            direction TB;
        style sg_m13 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m14["_BERTPredictionHeadTransform"]
            direction TB;
            style sg_m14 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            m15["Linear"];
            m16["LayerNorm"];
            end
            m17["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,8,768) → (1,8,30522)</span>"];
        end
        m18["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,768) → (1,2)</span>"];
        end
    end
    input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,8)</span>"];
    output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,8,30522)x2</span>"];
    style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
    style m3 fill:#f1f5f9,stroke:#475569,stroke-width:1px;
    style m4 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m5 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
    style m10 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m11 fill:#faf5ff,stroke:#6b46c1,stroke-width:1px;
    style m15 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m16 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    style m17 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    style m18 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
    input --> m3;
    m10 --> m11;
    m11 --> m15;
    m15 --> m16;
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

    class BERT(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*):
  Configuration object defining vocabulary, hidden dimensions, depth,
  attention setup, and runtime behavior (decoder/cache/pooling).

Methods
-------

.. automethod:: lucid.models.BERT.forward
.. automethod:: lucid.models.BERT.tie_weights
.. automethod:: lucid.models.BERT.get_input_embeddings
.. automethod:: lucid.models.BERT.set_input_embeddings
.. automethod:: lucid.models.BERT.get_output_embeddings
.. automethod:: lucid.models.BERT.set_output_embeddings

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.models as models
    >>> config = models.BERTConfig(
    ...     vocab_size=30522,
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     num_hidden_layers=12,
    ...     intermediate_size=3072,
    ...     hidden_act=lucid.nn.functional.gelu,
    ...     hidden_dropout_prob=0.1,
    ...     attention_probs_dropout_prob=0.1,
    ...     max_position_embeddings=512,
    ...     tie_word_embedding=True,
    ...     type_vocab_size=2,
    ...     initializer_range=0.02,
    ...     layer_norm_eps=1e-12,
    ...     use_cache=False,
    ...     is_decoder=False,
    ...     add_cross_attention=False,
    ...     chunk_size_feed_forward=0,
    ... )
    >>> model = models.BERT(config)
    >>> print(model)
    BERT(...)

End-to-End Example
------------------

.. code-block:: python

    >>> from pathlib import Path
    >>>
    >>> from lucid.models import BERTConfig, BERTTokenizerFast, BERTForPreTraining
    >>> from lucid.weights import BERT_Weights
    >>>
    >>> device = "gpu"
    >>>
    >>> # 1) Tokenizer build/save/reload
    >>> tokenizer = BERTTokenizerFast(vocab_file="some_vocab.txt")
    >>>
    >>> tokenizer.save_pretrained("some_path")
    >>> tokenizer = BERTTokenizerFast.from_pretrained("some_path")
    >>>
    >>> # 2) Model config + pretrained weights
    >>> config = BERTConfig(**BERT_Weights.DEFAULT.config)
    >>>
    >>> model = BERTForPreTraining(config)
    >>> model = model.from_pretrained(weights=BERT_Weights.DEFAULT)
    >>>
    >>> model.to(device)
    >>> model.eval()
    >>>
    >>> # 3) One-shot text-to-loss
    >>> loss = model.get_loss_from_text(
    ...     tokenizer=tokenizer,
    ...     text_a="Machine learning helps us build useful systems.",
    ...     text_b="Tokenization quality strongly affects language model performance.",
    ...     nsp_label=0,
    ...     device=device,
    ... )
    >>> print(loss.item())
    0.051273621783
