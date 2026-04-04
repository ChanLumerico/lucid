GPT2ForSequenceClassification
==============================

.. autoclass:: lucid.models.GPT2ForSequenceClassification

The `GPT2ForSequenceClassification` class applies a linear classification head
to the hidden state of the last non-padding token, following the GPT fine-tuning
approach for sequence-level tasks such as sentiment analysis and topic classification.

Class Signature
---------------

.. code-block:: python

    class GPT2ForSequenceClassification(config: GPT2Config, num_labels: int = 2)

Parameters
----------
- **config** (*GPT2Config*): GPT-2 configuration object.
- **num_labels** (*int*, optional): Number of target classes. Default is 2.

Methods
-------

.. automethod:: lucid.models.GPT2ForSequenceClassification.forward
   :no-index:

Compute sequence-level classification logits from the last non-padding token.
When `labels` are provided, also returns cross-entropy loss.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.GPT2Config.small()
    >>> model = models.GPT2ForSequenceClassification(config, num_labels=3)
    >>> print(model)
    GPT2ForSequenceClassification(...)

.. code-block:: python

    >>> import lucid
    >>> input_ids = lucid.randint(0, config.vocab_size, (4, 32))
    >>> labels = lucid.randint(0, 3, (4,))
    >>> loss, logits = model(input_ids, labels=labels)
    >>> logits.shape
    (4, 3)

.. code-block:: python

    >>> # Inference without labels
    >>> _, logits = model(input_ids)
    >>> preds = logits.argmax(axis=-1)
