BERTForSequenceClassification
=============================

.. autoclass:: lucid.models.BERTForSequenceClassification

The `BERTForSequenceClassification` class applies a classification head
to pooled BERT representations.

Class Signature
---------------

.. code-block:: python

    class BERTForSequenceClassification(config: BERTConfig, num_labels: int = 2)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration with pooling enabled.
- **num_labels** (*int*, optional): Number of target classes. Default is 2.

Methods
-------

.. automethod:: lucid.models.BERTForSequenceClassification.forward
   :no-index:

Compute sequence-level classification logits from pooled BERT features.

.. automethod:: lucid.models.BERTForSequenceClassification.get_loss
   :no-index:

Compute classification loss for sequence labels.

.. automethod:: lucid.models.BERTForSequenceClassification.predict_labels
   :no-index:

Return predicted sequence labels by argmax over logits.

.. automethod:: lucid.models.BERTForSequenceClassification.predict_proba
   :no-index:

Return class probabilities via softmax.

.. automethod:: lucid.models.BERTForSequenceClassification.get_accuracy
   :no-index:

Compute sequence classification accuracy.

.. automethod:: lucid.models.BERTForSequenceClassification.get_loss_from_text
   :no-index:

Compute sequence classification loss directly from raw text input.

.. automethod:: lucid.models.BERTForSequenceClassification.predict_labels_from_text
   :no-index:

Predict sequence labels directly from raw text input.

.. automethod:: lucid.models.BERTForSequenceClassification.predict_proba_from_text
   :no-index:

Predict class probabilities directly from raw text input.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_sequence_classification_base(num_labels=2)
    >>> print(model)
    BERTForSequenceClassification(...)

.. code-block:: python

    >>> logits = model(input_ids=input_ids, attention_mask=attention_mask)
    >>> loss = model.get_loss(labels=labels, input_ids=input_ids, attention_mask=attention_mask)
    >>> acc = model.get_accuracy(labels=labels, input_ids=input_ids, attention_mask=attention_mask)

.. code-block:: python

    >>> tokenizer = models.BERTTokenizerFast.from_pretrained(".data/bert/pretrained")
    >>> loss = model.get_loss_from_text(
    ...     tokenizer=tokenizer,
    ...     text_a="This movie is excellent.",
    ...     labels=1,
    ...     device="gpu",
    ... )
    >>> pred = model.predict_labels_from_text(
    ...     tokenizer=tokenizer,
    ...     text_a="This movie is excellent.",
    ...     device="gpu",
    ... )
