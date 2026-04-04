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

|transformer-badge| |decoder-only-transformer-badge|

.. autoclass:: lucid.models.GPT2

The `GPT2` class provides a decoder-only causal Transformer backbone following
the GPT-2 architecture. It extends GPT-1 with a final LayerNorm (`ln_f`)
applied after the decoder stack, a larger vocabulary (50257 tokens via Byte-level BPE),
a longer context window (1024 tokens).

.. mermaid::
    :name: GPT2

    %% Placeholder — fill in architecture diagram

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
