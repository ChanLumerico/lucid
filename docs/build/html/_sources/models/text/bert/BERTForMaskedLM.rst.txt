BERTForMaskedLM
===============

.. autoclass:: lucid.models.BERTForMaskedLM

The `BERTForMaskedLM` class attaches a masked language modeling head
to the BERT backbone.

Class Signature
---------------

.. code-block:: python

    class BERTForMaskedLM(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration for masked language modeling.

Methods
-------

.. automethod:: lucid.models.BERTForMaskedLM.forward
   :no-index:

Compute token logits over the vocabulary for each sequence position.

.. automethod:: lucid.models.BERTForMaskedLM.get_loss
   :no-index:

Compute masked language modeling loss from token labels.

.. automethod:: lucid.models.BERTForMaskedLM.create_masked_lm_inputs
   :no-index:

Build masked inputs and labels using the standard MLM masking policy
(mask/random/original replacements).

.. automethod:: lucid.models.BERTForMaskedLM.predict_token_ids
   :no-index:

Return argmax token predictions per position.

.. automethod:: lucid.models.BERTForMaskedLM.get_accuracy
   :no-index:

Compute token-level accuracy while ignoring masked-out label indices.

.. automethod:: lucid.models.BERTForMaskedLM.get_loss_from_text
   :no-index:

Compute MLM loss directly from raw text (with internal masking preparation).

.. automethod:: lucid.models.BERTForMaskedLM.predict_token_ids_from_text
   :no-index:

Predict token IDs directly from raw text input.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_masked_lm_base()
    >>> print(model)
    BERTForMaskedLM(...)

.. code-block:: python

    >>> masked_input_ids, labels = model.create_masked_lm_inputs(input_ids)
    >>> loss = model.get_loss(labels=labels, input_ids=masked_input_ids)
    >>> acc = model.get_accuracy(labels=labels, input_ids=masked_input_ids)

.. code-block:: python

    >>> tokenizer = models.BERTTokenizerFast.from_pretrained(".data/bert/pretrained")
    >>> loss = model.get_loss_from_text(
    ...     tokenizer=tokenizer,
    ...     text_a="Machine learning helps us build useful systems.",
    ...     text_b="Tokenization quality strongly affects language model performance.",
    ...     device="gpu",
    ... )
    >>> pred_ids = model.predict_token_ids_from_text(
    ...     tokenizer=tokenizer,
    ...     text_a="Machine learning is [MASK].",
    ...     device="gpu",
    ... )
