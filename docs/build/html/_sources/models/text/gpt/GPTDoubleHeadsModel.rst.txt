GPTDoubleHeadsModel
===================

.. autoclass:: lucid.models.GPTDoubleHeadsModel

The `GPTDoubleHeadsModel` class combines a causal language modeling head with
a multiple-choice classification head, following the original GPT-1 fine-tuning
approach for story-completion and commonsense reasoning tasks.
All answer choices are processed in parallel by folding the choice dimension
into the batch dimension.

Class Signature
---------------

.. code-block:: python

    class GPTDoubleHeadsModel(config: GPTConfig)

Parameters
----------
- **config** (*GPTConfig*): GPT configuration object.

Methods
-------

.. automethod:: lucid.models.GPTDoubleHeadsModel.forward
   :no-index:

Process multiple answer choices jointly and compute both language modeling
logits and multiple-choice logits. Returns
`(lm_loss, mc_loss, lm_logits, mc_logits, past_key_values)`.

.. automethod:: lucid.models.GPTDoubleHeadsModel.tie_weights
   :no-index:

Tie the `lm_head` projection weight to the input token embedding weight.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.GPTConfig.base()
    >>> model = models.GPTDoubleHeadsModel(config)
    >>> print(model)
    GPTDoubleHeadsModel(...)

.. code-block:: python

    >>> import lucid
    >>> # (batch=2, num_choices=4, seq_len=32)
    >>> input_ids = lucid.randint(0, config.vocab_size, (2, 4, 32))
    >>> mc_token_ids = lucid.randint(0, 32, (2, 4))  # EOS position per choice
    >>> mc_labels = lucid.randint(0, 4, (2,))         # correct choice index
    >>>
    >>> lm_loss, mc_loss, lm_logits, mc_logits, _ = model(
    ...     input_ids, mc_token_ids, mc_labels=mc_labels
    ... )
    >>> lm_logits.shape
    (2, 4, 32, 40478)
    >>> mc_logits.shape
    (2, 4)
