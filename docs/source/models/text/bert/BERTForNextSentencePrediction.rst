BERTForNextSentencePrediction
=============================

.. autoclass:: lucid.models.BERTForNextSentencePrediction

The `BERTForNextSentencePrediction` class applies the NSP head over
pooled BERT outputs.

Class Signature
---------------

.. code-block:: python

    class BERTForNextSentencePrediction(config: BERTConfig)

Parameters
----------
- **config** (*BERTConfig*): BERT configuration with pooling enabled.

Methods
-------

.. automethod:: lucid.models.BERTForNextSentencePrediction.forward
   :no-index:

Compute NSP logits from pooled BERT outputs.

.. automethod:: lucid.models.BERTForNextSentencePrediction.get_loss
   :no-index:

Compute next-sentence prediction classification loss.

.. automethod:: lucid.models.BERTForNextSentencePrediction.predict_labels
   :no-index:

Return predicted NSP labels by argmax over logits.

.. automethod:: lucid.models.BERTForNextSentencePrediction.predict_proba
   :no-index:

Return NSP class probabilities via softmax.

.. automethod:: lucid.models.BERTForNextSentencePrediction.get_accuracy
   :no-index:

Compute NSP classification accuracy.

.. automethod:: lucid.models.BERTForNextSentencePrediction.get_loss_from_text
   :no-index:

Compute NSP loss directly from text pairs.

.. automethod:: lucid.models.BERTForNextSentencePrediction.predict_labels_from_text
   :no-index:

Predict NSP labels directly from text pairs.

.. automethod:: lucid.models.BERTForNextSentencePrediction.predict_proba_from_text
   :no-index:

Predict NSP probabilities directly from text pairs.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.bert_for_next_sentence_prediction_base()
    >>> print(model)
    BERTForNextSentencePrediction(...)

.. code-block:: python

    >>> logits = model(input_ids=input_ids)
    >>> loss = model.get_loss(labels=labels, input_ids=input_ids)
    >>> acc = model.get_accuracy(labels=labels, input_ids=input_ids)

.. code-block:: python

    >>> tokenizer = models.BERTTokenizerFast.from_pretrained(".data/bert/pretrained")
    >>> loss = model.get_loss_from_text(
    ...     tokenizer=tokenizer,
    ...     text_a="A cat sits on the mat.",
    ...     text_b="It starts to rain outside.",
    ...     labels=0,
    ...     device="gpu",
    ... )
    >>> pred = model.predict_labels_from_text(
    ...     tokenizer=tokenizer,
    ...     text_a="A cat sits on the mat.",
    ...     text_b="It starts to rain outside.",
    ...     device="gpu",
    ... )
