GPTLMHeadModel
==============

.. autoclass:: lucid.models.GPTLMHeadModel

The `GPTLMHeadModel` class applies a linear language modeling head to the
GPT backbone for causal (autoregressive) next-token prediction.
The output projection weight is tied to the input token embedding.

Class Signature
---------------

.. code-block:: python

    class GPTLMHeadModel(config: GPTConfig)

Parameters
----------
- **config** (*GPTConfig*): GPT configuration object.

Methods
-------

.. automethod:: lucid.models.GPTLMHeadModel.forward
   :no-index:

Compute per-token logits over the vocabulary. When ``labels`` are provided,
also returns the shifted cross-entropy loss for next-token prediction.

.. automethod:: lucid.models.GPTLMHeadModel.tie_weights
   :no-index:

Tie the ``lm_head`` projection weight to the input token embedding weight.

.. automethod:: lucid.models.GPTLMHeadModel.get_input_embeddings
   :no-index:

Return the token embedding layer.

.. automethod:: lucid.models.GPTLMHeadModel.get_output_embeddings
   :no-index:

Return the ``lm_head`` linear projection.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.GPTConfig.base()
    >>> model = models.GPTLMHeadModel(config)
    >>> print(model)
    GPTLMHeadModel(...)

.. code-block:: python

    >>> import lucid
    >>> input_ids = lucid.randint(0, config.vocab_size, (2, 32))
    >>> loss, logits, _ = model(input_ids, labels=input_ids)
    >>> logits.shape
    (2, 32, 40478)

.. code-block:: python

    >>> # Greedy autoregressive generation
    >>> generated = input_ids
    >>> for _ in range(20):
    ...     _, logits, _ = model(generated)
    ...     next_token = logits[:, -1, :].argmax(axis=-1, keepdims=True)
    ...     generated = lucid.cat([generated, next_token], axis=-1)
