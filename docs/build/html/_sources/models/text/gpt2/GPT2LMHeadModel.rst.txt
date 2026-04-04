GPT2LMHeadModel
===============

.. autoclass:: lucid.models.GPT2LMHeadModel

The `GPT2LMHeadModel` class applies a linear language modeling head to the
GPT-2 backbone for causal (autoregressive) next-token prediction.
The output projection weight is tied to the input token embedding.

Class Signature
---------------

.. code-block:: python

    class GPT2LMHeadModel(config: GPT2Config)

Parameters
----------
- **config** (*GPT2Config*): GPT-2 configuration object.

Methods
-------

.. automethod:: lucid.models.GPT2LMHeadModel.forward
   :no-index:

Compute per-token logits over the vocabulary. When `labels` are provided,
also returns the shifted cross-entropy loss for next-token prediction.

.. automethod:: lucid.models.GPT2LMHeadModel.tie_weights
   :no-index:

Tie the `lm_head` projection weight to the input token embedding weight.

.. automethod:: lucid.models.GPT2LMHeadModel.get_input_embeddings
   :no-index:

Return the token embedding layer.

.. automethod:: lucid.models.GPT2LMHeadModel.get_output_embeddings
   :no-index:

Return the `lm_head` linear projection.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.GPT2Config.small()
    >>> model = models.GPT2LMHeadModel(config)
    >>> print(model)
    GPT2LMHeadModel(...)

.. code-block:: python

    >>> import lucid
    >>> input_ids = lucid.randint(0, config.vocab_size, (2, 32))
    >>> loss, logits, _ = model(input_ids, labels=input_ids)
    >>> logits.shape
    (2, 32, 50257)

.. code-block:: python

    >>> # Greedy autoregressive generation
    >>> generated = input_ids
    >>> for _ in range(20):
    ...     _, logits, _ = model(generated)
    ...     next_token = logits[:, -1, :].argmax(axis=-1, keepdims=True)
    ...     generated = lucid.cat([generated, next_token], axis=-1)
