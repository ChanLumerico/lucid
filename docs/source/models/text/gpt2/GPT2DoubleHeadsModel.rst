GPT2DoubleHeadsModel
====================

.. autoclass:: lucid.models.GPT2DoubleHeadsModel

The `GPT2DoubleHeadsModel` class combines a causal language modeling head with
a multiple-choice classification head, following the GPT fine-tuning approach
for story-completion and commonsense reasoning tasks.
All answer choices are processed in parallel by folding the choice dimension
into the batch dimension.

Class Signature
---------------

.. code-block:: python

    class GPT2DoubleHeadsModel(config: GPT2Config)

Parameters
----------
- **config** (*GPT2Config*): GPT-2 configuration object.

Methods
-------

.. automethod:: lucid.models.GPT2DoubleHeadsModel.forward
   :no-index:

Process multiple answer choices jointly and compute both language modeling
logits and multiple-choice logits. Returns
`(lm_loss, mc_loss, lm_logits, mc_logits, past_key_values)`.

.. automethod:: lucid.models.GPT2DoubleHeadsModel.tie_weights
   :no-index:

Tie the `lm_head` projection weight to the input token embedding weight.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.GPT2Config.small()
    >>> model = models.GPT2DoubleHeadsModel(config)
    >>> print(model)
    GPT2DoubleHeadsModel(...)

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
    (2, 4, 32, 50257)
    >>> mc_logits.shape
    (2, 4)
