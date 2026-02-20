BERTForCausalLM
===============

.. autoclass:: lucid.models.BERTForCausalLM

The `BERTForCausalLM` class configures BERT in decoder-style mode and
applies a causal language modeling head.

Class Signature
---------------

.. code-block:: python

    class BERTForCausalLM(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration with decoder/caching options.

Methods
-------

.. automethod:: lucid.models.BERTForCausalLM.forward
   :no-index:

Compute autoregressive token logits from decoder-style BERT outputs.

.. automethod:: lucid.models.BERTForCausalLM.get_loss
   :no-index:

Compute causal language modeling loss (with optional label shifting).

.. automethod:: lucid.models.BERTForCausalLM.predict_token_ids
   :no-index:

Return argmax token IDs for each step in the current sequence.

.. automethod:: lucid.models.BERTForCausalLM.get_accuracy
   :no-index:

Compute token-level accuracy for causal LM targets.

.. automethod:: lucid.models.BERTForCausalLM.get_perplexity
   :no-index:

Compute perplexity from mean causal LM loss.

.. automethod:: lucid.models.BERTForCausalLM.get_next_token_logits
   :no-index:

Return logits for the next token (last position only).

.. automethod:: lucid.models.BERTForCausalLM.predict_next_token_id
   :no-index:

Return argmax next-token IDs from last-position logits.

.. automethod:: lucid.models.BERTForCausalLM.get_loss_from_text
   :no-index:

Compute causal LM loss directly from raw text input.

.. automethod:: lucid.models.BERTForCausalLM.predict_next_token_id_from_text
   :no-index:

Predict next-token IDs directly from raw text input.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_causal_lm_base()
    >>> print(model)
    BERTForCausalLM(...)

.. code-block:: python

    >>> loss = model.get_loss(labels=input_ids, input_ids=input_ids, shift_labels=True)
    >>> ppl = model.get_perplexity(labels=input_ids, input_ids=input_ids)
    >>> next_ids = model.predict_next_token_id(input_ids=input_ids)

.. code-block:: python

    >>> tokenizer = models.BERTTokenizerFast.from_pretrained(".data/bert/pretrained")
    >>> loss = model.get_loss_from_text(
    ...     tokenizer=tokenizer,
    ...     text_a="Language models predict the next token.",
    ...     device="gpu",
    ... )
    >>> next_ids = model.predict_next_token_id_from_text(
    ...     tokenizer=tokenizer,
    ...     text_a="Deep learning is",
    ...     device="gpu",
    ... )
